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
    default_model_id: str = Field(default="openai/gpt-oss-20b")
    gpt_oss_model_id: str = Field(default="openai/gpt-oss-20b")
    solar_model_id: str = Field(default="upstage/SOLAR-10.7B-Instruct-v1.0")
    eeve_model_id: str = Field(default="yanolja/YanoljaNEXT-EEVE-Instruct-10.8B")

    embed_model_name: str = Field(default="jhgan/ko-sbert-nli")
    base_index_dir: str = Field(default="data/story_indexes")

    class Config:
        env_prefix = "MLCHAT_"
        case_sensitive = False

@lru_cache()
def get_settings() -> Settings:
    return Settings()  # type: ignore[arg-type]

settings = get_settings()

__all__ = ["settings", "get_settings", "Settings"]
