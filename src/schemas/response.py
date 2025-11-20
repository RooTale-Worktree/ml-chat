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
    finish_reason: Optional[FinishReason] = None


# LLM 응답을 관리: 자연어와 embedding 결과 2개를 반환
class ResponseContent(BaseModel):
    content: str                            # 생성 텍스트
    embedding: Optional[List[float]] = None # 텍스트 embedding


# 실행 모델 메타 데이터
class ModelInfo(BaseModel):
    name: Optional[str] = None              # 실제 사용 모델명
    context_length: Optional[int] = None
    embedding_model: Optional[str] = None   # 임베딩에 사용한 모델명/버전
    dtype: Optional[str] = None


# handler가 반환해야하는 response 형식
class ChatResponse(BaseModel):
    session_id: str
    responded_as: Literal["narrator", "character"] = "character"
    response_contents: List[ResponseContent]                   # 보통 1개, 샘플링/beam 시 N개 가능
    usage: Optional[Usage] = None

    # 실행/디버깅 메타
    model_info: Optional[ModelInfo] = None
    timing: Optional[Dict[str, Any]] = None
    error: Optional[str] = None