## 개요

스토리 기반 context와 최근 대화 맥락을 결합하고, Persona 정보를 포함한 프롬프트를 구성하여 캐릭터 특화 LLM에게 질의하는 구조의 Chat 서비스 스캐폴딩입니다.

## 디렉토리 구조

```
.
├── data
│   ├── indexes
│   └── mock
│       ├── sample_history.json
│       ├── sample_persona.json
│       ├── sample_request.json
│       ├── sample_story.json
│       └── vector
│           ├── sha1:0c26aa7b21f177658276a420c390090638af6a1e.npy
│           └── sha1:6a5009a8cd7490c2121130087542b613a7cf3a7d.npy
├── Dockerfile
├── handler.py
├── README.md
├── requirements.txt
├── scripts
│   ├── build_index.py
│   └── embed_history.py
├── src
│   ├── config
│   │   └── config.py
│   ├── core
│   │   ├── embedding.py
│   │   └── prompt_builder.py
│   ├── llm
│   │   ├── mock_llm.py
│   │   └── pygmalion_llm.py
│   ├── orchestrator.py
│   ├── rag
│   │   ├── chat_rag.py
│   │   └── story_rag.py
│   ├── schemas
│   │   ├── embedding.py
│   │   ├── rag.py
│   │   ├── request.py
│   │   └── response.py
│   └── utils
└── tests
    └── test_prompt_builder.py
```

## 스크립트 기능

1. `handler.py`
    - Runpod Stateless의 entrypoint입니다.
    - 내부에 있는 `handler(event)` 함수를 호출하면 `event[’input’]`을 파싱하여 `src/orchestrator.py`의 `handle_chat(payload)` 함수를 호출합니다.
    - root 디렉토리에서 shell에 아래 명령어를 입력하여 로컬 테스트를 진행할 수 있습니다.
        
        ```bash
        python -m handler
        ```
        
    - 로컬 테스트 시에 CLI 옵션을 통해 어떤 mock-up data를 불러올지 지정할 수 있습니다. 자세한 옵션은 `handler.py`의 `argparser`를 통해 확인할 수 있습니다.
        
        ```bash
        python -m handler --persona ./data/mock/sample_persona.json
        ```
        
2. `src/orchestrator.py`
    - 한 턴의 대화에 대해 ‘사용자 질의’ → ‘LLM 응답’의 전체 과정을 관리하기 위한 스크립트입니다.
    - 다음의 workflow로 한 턴의 대화를 처리합니다.
        1. `src/core/embedding.py`의 `embed_text(text)`를 호출하여 사용자의 질의문을 벡터로 임베딩합니다. (RAG에서 cosine 유사도를 계산하기 위함)
        2. `src/rag/chat_rag.py`의 `retrieve_chat_context()`함수를 호출하여 지난 채팅 맥락 기반 RAG를 호출합니다.
        3. `src/rag/story_rag.py`의 `retrieve_story_context()`함수를 호출하여 스토리 기반 RAG를 호출합니다.
        4. `src/core/prompt_builder.py`의 `build_prompt()`함수를 호출합니다. persona, chat RAG, story RAG, 사용자 질의를 기반으로 프롬프트를 생성합니다.
        5. LLM 모델을 로드한 후 답변을 생성합니다.
        6. Response 타입에 맞게 파싱한 후, 결과를 반환합니다.
3. `src/schemas/request.py`
    - ‘사용자 질의’의 형식을 관리하기 위한 pydantic model을 정의합니다. (백엔드 → serverless 호출에서의 interface)
    - 채팅 히스토리, RAG 조건, 스토리, 사용 모델 등의 정보를 관리합니다.
        
        ```python
        # handler가 처리해야하는 request의 형식
        class ChatRequest(BaseModel):
            message: str
            session_id: Optional[str] = None
            user_id: Optional[str] = None
            persona: Persona
            history: List[Message] = Field(default_factory=list)
            chat_rag_config: ChatRAGConfig = Field(default_factory=ChatRAGConfig)
            story: List[StoryEvent] = Field(default_factory=list)
            model: ModelConfig = Field(default_factory=ModelConfig)
            gen: GenConfig = Field(default_factory=GenConfig)
            meta: Dict[str, Any] = Field(default_factory=dict)
        ```
        
4. `src/schemas/response.py`
    - ‘LLM 응답’의 형식을 관리하기 위한 pydantic model을 정의합니다. (serverless → 백엔드 응답에서의 interface)
    - LLM 응답, RAG 점수, 사용 프롬프트 등의 정보를 관리합니다.
        
        ```python
        # handler가 반환해야하는 response 형식
        class ChatResponse(BaseModel):
            session_id: str
            responded_as: Literal["narrator", "character"] = "character"
            response_contents: List[ResponseContent]
            usage: Optional[Usage] = None
            retrieved: List[RetrievalItem] = Field(default_factory=list)
        
            # 실행/디버깅 메타
            model_info: Optional[ModelInfo] = None
            timing: Optional[Timing] = None
            error: Optional[str] = None
            meta: Dict[str, Any] = Field(default_factory=dict)
        ```
        
5. `src/schemas/rag.py`
    - Serverless 내부에서 RAG 및 prompt의 input과 output을 관리하기 위한 pydantic model을 정의합니다.
    - 아직 RAG 부분의 구현이 완벽하지 않으므로 수정될 수 있습니다.
        
        ```python
        class RAGChunk(BaseModel):
            id: str
            source: str
            text: str
            score: Optional[float] = None
            meta: Dict[str, Any] = {}
        
        class ChatRAGResult(BaseModel):
            context: List[RAGChunk]
        
        class StoryRAGResult(BaseModel):
            context: List[RAGChunk]
        
        class PromptBuildInput(BaseModel):
            persona: Persona
            chat_context: List[RAGChunk]
            story_context: List[RAGChunk]
            history: List[DialogueTurn]
            user_message: str
        
        class PromptBuildOutput(BaseModel):
            prompt: str
            meta: Dict[str, Any] = {}
        ```
        

## TODO

- `src/rag/chat_rag.py`, `src/rag/story_rag.py` 구현
- LLM 모델 선정
- 채팅 context를 유지하기 위해 어떤 정보들을 RAG에 활용할 수 있을지
- 단순 채팅이 아니라 스토리 진행도 함께: Pygmalion의 `narrator` role을 활용할 수 있을지