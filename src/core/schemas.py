from __future__ import annotations
"""Pydantic schemas for request/response + internal domain objects."""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class DialogueTurn(BaseModel):
    role: str = Field(description="user|character")
    text: str

class Persona(BaseModel):
    name: str
    persona: str
    scenario: str
    speaking_style: str
    constraints: List[str] = []
    example_dialogue: List[DialogueTurn] = []

class RAGChunk(BaseModel):
    id: str
    source: str
    text: str
    score: Optional[float] = None
    meta: Dict[str, Any] = {}

class ChatRequest(BaseModel):
    message: str
    history: List[DialogueTurn] = []
    persona: Persona
    gen: Dict[str, Any] = {}
    model: Dict[str, Any] = {}

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

class ChatResponse(BaseModel):
    reply: str
    prompt: str
    chat_context: List[RAGChunk]
    story_context: List[RAGChunk]
    meta: Dict[str, Any] = {}

__all__ = [
    "DialogueTurn","Persona","RAGChunk","ChatRequest","ChatRAGResult","StoryRAGResult",
    "PromptBuildInput","PromptBuildOutput","ChatResponse"
]
