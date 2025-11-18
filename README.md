## 개요

캐릭터 persona, 스토리 맥락, 대화 기록을 결합하여 일관된 캐릭터 응답을 생성하는 LLM 기반 채팅 서비스입니다.
RAG(Retrieval-Augmented Generation)를 활용해 관련 컨텍스트를 검색하고, vLLM으로 최적화된 추론을 제공합니다.

**주요 기능:**
- 캐릭터별 persona 기반 roleplay 대화
- 대화 기록 및 스토리 컨텍스트 RAG
- 다중 LLM 모델 지원 (GPT-OSS, Solar, EEVE)
- RunPod Serverless 배포 지원
- vLLM 기반 고성능 추론

## 디렉토리 구조

```
.
├── data
│   ├── indexes
│   └── mock
│       ├── sample_chat_history.json
│       ├── sample_persona_luffy.json
│       ├── sample_persona.json
│       ├── sample_request_gpt_oss.json
│       ├── sample_request_solar.json
│       ├── sample_request_storyrag_test.json
│       └── sample_story.json
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
│   │   ├── build_scene_index.py
│   │   └── embedding.py
│   ├── fine_tune
│   │   └── gpt_20b_tune.py
│   ├── llm
│   │   ├── eeve_llm.py
│   │   ├── get_llm.py
│   │   ├── gpt_oss_llm.py
│   │   └── solar_llm.py
│   ├── orchestrator.py
│   ├── prompt
│   │   ├── eeve_prompt.py
│   │   ├── get_prompt.py
│   │   ├── gpt_oss_prompt.py
│   │   └── solar_prompt.py
│   ├── rag
│   │   ├── chat_rag.py
│   │   └── story_rag.py
│   └── schemas
│       ├── rag.py
│       ├── request.py
│       └── response.py
└── tests
    └── test_prompt_builder.py
```

## 스크립트 기능

### 1. `handler.py`
RunPod Serverless의 entrypoint입니다.
- `handler(event)` 함수가 요청을 받아 `orchestrator.py`의 `handle_chat()`을 호출
- 로컬 테스트 지원: `python -m handler`

**CLI 옵션:**
```bash
python -m handler \
  --persona "./data/mock/sample_persona.json" \
  --chat_history "./data/mock/sample_chat_history.json" \
  --story "./data/mock/sample_story.json" \
  --others "./data/mock/sample_request_gpt.json"
  --message "안녕, 오늘 기분 어때?"
```

### 2. `src/orchestrator.py`
한 턴의 대화 처리를 총괄하는 오케스트레이터입니다.

**처리 흐름:**
1. 사용자 질의를 임베딩 (`src/core/embedding.py`)
2. 대화 기록 RAG 검색 (`src/rag/chat_rag.py`)
3. 스토리 컨텍스트 RAG 검색 (`src/rag/story_rag.py`)
4. Persona + RAG 결과로 프롬프트 구성 (`src/prompt/get_prompt.py`)
5. LLM 모델 로드 및 응답 생성 (`src/llm/get_llm.py`)
6. 응답 임베딩 및 메타데이터 포함하여 반환

### 3. `src/schemas/`
Pydantic 모델로 요청/응답 스키마를 정의합니다.

**`request.py`:**
- `ChatRequest`: 사용자 요청 전체 (persona, history, story, model config 등)
- `Persona`: 캐릭터 정보 (이름, 성격, 시나리오, 예시 대화)
- `Message`: 대화 메시지 (role, content, embedding)
- `ModelConfig`: 모델 설정 (name, tensor_parallel_size, gpu_memory_utilization)
- `GenConfig`: 생성 하이퍼파라미터 (temperature, top_p, max_new_tokens)

**`response.py`:**
- `ChatResponse`: LLM 응답 전체 (reply, usage, retrieved context, timing)
- `ResponseContent`: 생성된 텍스트와 임베딩
- `Usage`: 토큰 사용량
- `Timing`: 각 단계별 처리 시간 (ms)

**`rag.py`:**
- `RAGChunk`: 검색된 컨텍스트 조각 (text, score, source)
- `PromptBuildInput`: 프롬프트 빌더 입력
- `PromptBuildOutput`: 완성된 프롬프트 텍스트

### 4. `src/llm/`
LLM 모델 인터페이스를 제공합니다.

**`get_llm.py`:**
- 모델 이름에 따라 적절한 LLM 인스턴스 반환
- 지원 모델: `gpt-oss-20b`, `solar-10.7b`, `eeve-10.8b`

**`gpt_oss_llm.py`:**
- vLLM 기반 GPT-OSS 20B 모델 래퍼
- PagedAttention, continuous batching으로 최적화
- `generate(prompt, **gen)` → `{"reply": str, "usage": dict, "raw": str}`

**`solar_llm.py` / `eeve_llm.py`:**
- 각 모델별 특화 래퍼 (transformers 기반)

### 5. `src/prompt/`
모델별로 최적화된 프롬프트 템플릿을 제공합니다.

**`get_prompt.py`:**
- 모델 이름에 따라 적절한 프롬프트 빌더 선택

**`gpt_oss_prompt.py`:**
- GPT-OSS용 단순 텍스트 기반 프롬프트
- Persona, 예시 대화, RAG 컨텍스트, 대화 기록을 조합

**`solar_prompt.py` / `eeve_prompt.py`:**
- 각 모델의 chat template에 맞춘 프롬프트 생성

### 6. `src/rag/`
RAG 컨텍스트 검색 모듈입니다.

**`chat_rag.py`:**
- 대화 기록에서 유사한 과거 대화 검색
- 코사인 유사도 기반, top-k 필터링

**`story_rag.py`:**
- 스토리 이벤트에서 관련 맥락 검색
- 시간 윈도우 및 유사도 기반 필터링

### 7. `src/core/`
임베딩 및 인덱스 빌드 유틸리티입니다.

**`embedding.py`:**
- `sentence-transformers`로 텍스트 임베딩 생성
- 기본 모델: `paraphrase-multilingual-MiniLM-L12-v2`

**`build_scene_index.py`:**
- 스토리 장면 벡터 인덱스 구축

### 8. `src/config/config.py`
환경 변수 및 전역 설정을 관리합니다.
- 모델 ID, 디폴트 하이퍼파라미터
- RAG top-k 설정
- 데이터 디렉토리 경로

### 9. `scripts/`
데이터 전처리 및 인덱스 빌드 스크립트입니다.

**`build_index.py`:**
- 대화 기록 벡터 인덱스 생성

**`embed_history.py`:**
- 대화 메시지 임베딩 일괄 생성

### 10. `data/mock/`
로컬 테스트용 샘플 데이터입니다.
- `sample_persona_*.json`: 캐릭터 persona 예시
- `sample_chat_history.json`: 대화 기록 예시
- `sample_story.json`: 스토리 이벤트 예시
- `sample_request_*.json`: 요청 페이로드 예시

## 로컬 실행

### 1. 환경 설정
```bash
# 가상환경 생성 및 활성화
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate  # Windows

# 의존성 설치
pip install -r requirements.txt
```

### 2. 로컬 테스트
```bash
# 기본 실행
python -m handler

# 커스텀 persona로 실행
python -m handler \
  --persona ./data/mock/sample_persona_luffy.json \
  --message "고기 먹으러 갈래?"

# 특정 모델로 실행
python -m handler \
  --others ./data/mock/sample_request_gpt_oss.json \
  --message "오늘 뭐 했어?"
```

### 3. 출력 확인
```
[Model Prompt]
--- 대화 시뮬레이션 ---
당신은 '루피'입니다.
...

[Model Reply]
오~ 고기! 좋지! 언제 가? 지금 당장?!

[Timing ms]
{
  "total_ms": 1234,
  "llm_load_ms": 0,
  "generate_ms": 850,
  ...
}
```

## RunPod Serverless 배포

### 1. Docker 이미지 빌드
```bash
docker build -t your-registry/chat-service:latest .
docker push your-registry/chat-service:latest
```

### 2. RunPod 설정
- **Image**: `your-registry/chat-service:latest`
- **GPU**: A100 40GB 이상 권장 (gpt-oss-20b 기준)
- **Environment Variables**:
  ```
  MLCHAT_GPT_OSS_MODEL_ID=openai/gpt-oss-20b
  MLCHAT_ENV=production
  ```

### 3. 성능 최적화
- `tensor_parallel_size=2`: 멀티 GPU 병렬화
- `gpu_memory_utilization=0.9`: GPU 메모리 사용률
- `max_model_len=2048`: 최대 시퀀스 길이

### 4. Cold Start 최적화
- 모델은 컨테이너 시작 시 전역에서 1회 로딩
- Warm container 재사용으로 지연 시간 최소화

## 지원 모델

| 모델 | 크기 | 특징 | vLLM 지원 |
|------|------|------|-----------|
| GPT-OSS | 20B | 베이스 모델, 높은 자유도 | ✅ |
| Solar | 10.7B | 한국어 특화, 빠른 추론 | ✅ |
| EEVE | 10.8B | 일상 대화 최적화 | ✅ |

## 성능 벤치마크

**환경**: RTX 5090 (24GB), vLLM, max_new_tokens=256

| 모델 | Cold Start | Warm Generate | Throughput |
|------|-----------|---------------|------------|
| GPT-OSS 20B | ~30s | ~1.2s | 200 tok/s |
| Solar 10.7B | ~15s | ~0.6s | ??? tok/s |
| EEVE 10.8B | ~15s | ~0.6s | ??? tok/s |

## TODO

- [ ] 대화 기록 벡터 DB 연동 (ChromaDB/Qdrant)
- [ ] 스토리 RAG 가중치 튜닝
- [ ] 멀티턴 대화 컨텍스트 윈도우 최적화
- [ ] Narrator 역할 분리 및 스토리 진행 로직
- [ ] 감정 분석 및 호감도 동적 업데이트
- [ ] 응답 품질 평가 메트릭 (coherence, consistency)

## 라이선스

MIT License