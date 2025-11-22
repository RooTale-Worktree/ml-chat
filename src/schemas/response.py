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
    narrative: str
    character_message: str
    embedding: Optional[List[float]] = None


class Timing(BaseModel):
    message_embed_ms: Optional[int] = None
    chat_retr_ms: Optional[int] = None
    story_retr_ms: Optional[int] = None
    llm_load_ms: int
    generate_ms: int
    response_embed_ms: Optional[int] = None
    total_ms: int


# handler가 반환해야하는 response 형식
class ChatResponse(BaseModel):
    responded_as: Literal["narrator", "character"] = "character"
    response_contents: ResponseContent
    usage: Optional[Usage] = None
    timing: Optional[Timing] = None