"""
Pydantic schemas for RAG objects.
"""
from __future__ import annotations
from pydantic import BaseModel, UUID4, Field
from typing import List, Optional, Dict, Any, Literal
from datetime import datetime

from src.schemas.request import Persona, Message, DialogueTurn


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
    chat_context: List[Message]
    story_context: List[RAGChunk]
    user_message: str


class PromptBuildOutput(BaseModel):
    prompt: str
    meta: Dict[str, Any] = {}