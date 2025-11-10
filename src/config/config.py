"""
Central configuration loader.

Environment precedence:
1. Explicit function arguments / overrides
2. Environment variables
3. Defaults in code

Add any service-wide constants or dynamic settings here.
"""
from __future__ import annotations
from pydantic import Field
from pydantic_settings import BaseSettings
from functools import lru_cache
import os

class Settings(BaseSettings):
    env: str = Field(default="local")
    log_level: str = Field(default="INFO")

    # Model / inference
    default_model_id: str = Field(default="PygmalionAI/pygmalion-6b")
    gpt_oss_model_id: str = Field(default="openai-community/gpt-oss-20b")
    max_new_tokens: int = 256
    temperature: float = 0.8
    top_p: float = 0.9
    repetition_penalty: float = 1.1

    # RAG / retrieval params (stub values)
    chat_k: int = 4
    story_k: int = 5

    data_dir: str = Field(default="data")
    index_dir: str = Field(default="data/indexes")

    class Config:
        env_prefix = "MLCHAT_"
        case_sensitive = False

@lru_cache()
def get_settings() -> Settings:
    return Settings()  # type: ignore[arg-type]

settings = get_settings()

__all__ = ["settings", "get_settings", "Settings"]
