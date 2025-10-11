"""
Pydantic schemas for response objects
"""
from __future__ import annotations
from pydantic import BaseModel, UUID4, Field
from typing import List, Optional, Dict, Any, Literal
from datetime import datetime


Role = Literal["user", "assistant", "system", "character", "narrator"]
FinishReason = Literal["stop", "length", "eos", "content_filter", "error"]

# 답변 생성에서의 토큰 사용량을 나타내는 메타 데이터
class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

# RAG 단계에서 프롬프트에 넣은 컨텍스트 조각의 목록과 점수
class RetrievalItem(BaseModel):
    source: Literal["history", "story"]
    id: Optional[str] = None
    role: Optional[Role] = None             # 원본 메시지의 역할
    content: str
    score: float
    rank: Optional[int] = None              # 몇 등으로 뽑혔는지
    meta: Dict[str, Any] = Field(default_factory=dict)

# LLM 응답을 관리: 자연어와 embedding 결과 2개를 반환
class ResponseContent(BaseModel):
    role: Role = "character"                # "narrator"도 가능
    content: str                            # 생성 텍스트
    embedding: Optional[List[float]] = None # 텍스트 embedding
    character_id: Optional[UUID4] = None    # role이 "character"일 때 연결된 캐릭터
    character_name: Optional[str] = None
    finish_reason: Optional[FinishReason] = None
    safety_labels: Dict[str, Any] = Field(default_factory=dict) # 콘텐츠 필터링 관련 메타

# 실행 모델 메타 데이터
class ModelInfo(BaseModel):
    provider: Optional[str] = None
    name: Optional[str] = None              # 실제 사용 모델명
    context_length: Optional[int] = None
    embedding_model: Optional[str] = None   # 임베딩에 사용한 모델명/버전
    dtype: Optional[str] = None

# 응답 생성 메타 데이터
class Timing(BaseModel):
    queued_ms: Optional[int] = None
    embed_ms: Optional[int] = None
    retrieve_ms: Optional[int] = None
    prompt_build_ms: Optional[int] = None
    generate_ms: Optional[int] = None
    total_ms: Optional[int] = None          # LLM 응답에서 부적절 또는 금칙어 등이 포함되어 있는지 여부

# handler가 반환해야하는 response 형식
class ChatResponse(BaseModel):
    session_id: str
    responded_as: Literal["narrator", "character"] = "character"
    response_contents: List[ResponseContent]                   # 보통 1개, 샘플링/beam 시 N개 가능
    usage: Optional[Usage] = None
    retrieved: List[RetrievalItem] = Field(default_factory=list)

    # 실행/디버깅 메타
    model_info: Optional[ModelInfo] = None
    timing: Optional[Timing] = None
    error: Optional[str] = None
    meta: Dict[str, Any] = Field(default_factory=dict)