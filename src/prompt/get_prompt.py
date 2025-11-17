"""
Prompt dispatcher that selects a model-specific template.
"""
from __future__ import annotations
from typing import Callable, Dict, Iterable

from src.schemas.rag import PromptBuildInput, PromptBuildOutput
from src.prompt.pygmalion_prompt import build_prompt as build_pygmalion_prompt
from src.prompt.gpt_oss_prompt import build_prompt as build_gpt_oss_prompt

PromptBuilder = Callable[[PromptBuildInput], PromptBuildOutput]


def get_prompt(model_name: str | None, data: PromptBuildInput) -> PromptBuildOutput:
    """
    Resolve the proper prompt builder for the requested model.
    """
    if model_name is None or model_name == "mock_llm":
        return None
    
    elif model_name == "pygmalion-6b":
        return build_pygmalion_prompt(data)
    
    elif model_name == "gpt-oss-20b":
        return build_gpt_oss_prompt(data)
    
    raise ValueError(f"Not supported model for prompt building: {model_name}")


__all__ = ["get_prompt", "PromptBuilder"]
