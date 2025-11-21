"""
Pydantic schemas for request objects.
"""
from __future__ import annotations
from pydantic import BaseModel, UUID4, Field
from typing import List, Optional, Dict, Any, Literal
from datetime import datetime


Role = Literal["user", "assistant", "system", "character", "narrator"]


# Manage example dialogues within persona
class DialogueTurn(BaseModel):
    role: Role
    content: str


# Manage character persona information
class Persona(BaseModel):
    character_name: str
    persona: str
    scenario: str
    speaking_style: List[str] = Field(default_factory=list)
    constraints: List[str] = Field(default_factory=list)
    example_dialogue: List[DialogueTurn] = Field(default_factory=list)
    meta: Optional[Dict[str, Any]] = Field(default_factory=dict)


# Manage user questions, system answers, and prompts collectively
class Message(BaseModel):
    content: str
    role: Role
    embedding: Optional[List[float]] = None
    embedding_dim: Optional[int] = None
    embedding_model: Optional[str] = None
    embedding_etag: Optional[str] = None
    meta: Optional[Dict[str, Any]] = Field(default_factory=dict)

    def to_dialogue_turn(self) -> DialogueTurn:
        return DialogueTurn(
            role=self.role,
            content=self.content,
        )


# Configurations for chat-based RAG
class ChatRAGConfig(BaseModel):
    top_k_history: int = 6
    history_time_window_min: Optional[int] = None
    measure: Optional[str] = Field(default="cosine")
    threshold: Optional[float] = Field(default=0.12)
    meta: Optional[Dict[str, Any]] = Field(default_factory=dict)


class ModelConfig(BaseModel):
    model_name: str = "gpt-oss-20b"
    tensor_parallel_size: int = 1
    gpu_memory_utilization: Optional[float] = 0.9
    max_model_len: Optional[int] = 131_072
    max_num_seqs: Optional[int] = 16
    trust_remote_code: bool = True
    dtype: Optional[str] = "auto"
    meta: Optional[Dict[str, Any]] = Field(default_factory=dict)


# Hyperparameters related to LLM generation sampling
class GenConfig(BaseModel):
    temperature: float = 0.8
    top_p: float = 0.95
    max_new_tokens: int = 256
    repetition_penalty: Optional[float] = 1.05
    stop: Optional[List[str]] = Field(default_factory=list)
    reasoning_effort: Literal["low", "medium", "high"] = "medium"


# Format of the request that the handler should process
class ChatRequest(BaseModel):
    message: str
    user_name: Optional[str] = None
    persona: Persona
    chat_history: Optional[List[Message]] = Field(default_factory=list)
    chat_rag_config: Optional[ChatRAGConfig] = None
    story_title: Optional[str] = None
    model_config: ModelConfig = Field(default_factory=ModelConfig)
    gen: GenConfig = Field(default_factory=GenConfig)
    meta: Optional[Dict[str, Any]] = Field(default_factory=dict)