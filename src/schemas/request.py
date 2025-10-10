"""
Pydantic schemas for request objects.
"""
from __future__ import annotations
from pydantic import BaseModel, UUID4, Field
from typing import List, Optional, Dict, Any, Literal
from datetime import datetime


Role = Literal["user", "assistant", "system", "character", "narrator"]

# persona에서 예시 대화를 관리하기 위한 table
class DialogueTurn(BaseModel):
    role: Role
    content: str
    character_id: Optional[UUID4] = None
    timestamp: Optional[datetime] = None

# 캐릭터의 persona를 관리
class Persona(BaseModel):
    persona_id: UUID4
    name: str
    persona: str
    scenario: str
    speaking_style: str
    constraints: List[str] = Field(default_factory=list)
    example_dialogue: List[DialogueTurn] = Field(default_factory=list)
    meta: Dict[str, Any] = Field(default_factory=dict)

# 사용자 질문, 시스템 답변, 프롬프트를 통합적으로 관리
class Message(BaseModel):
    chat_id: UUID4
    seq_no: int
    role: Role
    content: str
    character_id: Optional[UUID4] = None
    character_name: Optional[str] = None
    timestamp: Optional[datetime] = None
    tokens: Optional[int] = None
    meta: Dict[str, Any] = Field(default_factory=dict)
    # optional: embedding을 보내는 경우
    embedding: Optional[List[float]] = None
    embedding_dim: Optional[int] = None
    embedding_model: Optional[str] = None
    embedding_etag: Optional[str] = None

# 채팅 기반 RAG를 위한 config
class ChatRAGConfig(BaseModel):
    top_k_history: int = 6
    history_time_window_min: Optional[int] = None
    min_cosine: float = 0.12
    allow_compute_missing: bool = False     # embedding이 비어있는 table이 있는 경우 계산

# 스토리 RAG를 위해 스토리 정보를 입력으로 받음
class StoryEvent(BaseModel):
    story_id: Optional[UUID4] = None
    story_type: str = "event"
    content: str
    timestamp: Optional[datetime] = None
    meta: Dict[str, Any] = Field(default_factory=dict)
    # optional: embedding을 보내는 경우
    embedding: Optional[List[float]] = None
    embedding_dim: Optional[int] = None
    embedding_model: Optional[str] = None
    embedding_etag: Optional[str] = None

# 응답에 사용할 모델과 관련된 엔진/리소스를 설정
class ModelConfig(BaseModel):
    name: str = "pygmalion-6b"
    context_length: int = 4096
    device: str = "auto"         # "cpu", "cuda", "mps", "auto"
    dtype: Optional[str] = None  # "fp16", "fp32", "bf16" 등

# LLM 생성 샘플링 관련 하이퍼파라미터
class GenConfig(BaseModel):
    temperature: float = 0.8
    top_p: float = 0.95
    max_new_tokens: int = 256
    repetition_penalty: Optional[float] = 1.05 
    stop: List[str] = Field(default_factory=list)

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