"""
Central place to load LLM instances depending on requested model.
"""
from __future__ import annotations
from typing import Any, Dict, Iterable

from src.config.config import settings
from src.llm.mock_llm import MockLLM
from src.llm.pygmalion_llm import PygmalionLLM
from src.llm.gpt_oss_llm import load_gpt_oss_llm
from src.schemas.request import ModelConfig

_LLM_CACHE: Dict[str, Any] = {}


def _device_map(model_cfg: ModelConfig | None) -> str | None:
    if model_cfg and getattr(model_cfg, "device", None):
        device = model_cfg.device
        if device and device.lower() != "auto":
            return device
    return "auto"


def get_llm(model_name: str | None, model_cfg: ModelConfig | None = None):
    """
    Return a cached LLM instance for the requested model.
    """
    if model_name is None or model_name == "mock_llm":
        return MockLLM()
    
    elif model_name == "pygmalion-6b":
        device_map = _device_map(model_cfg)
        cache_key = f"pygmalion::{settings.default_model_id}::{device_map}"
        if cache_key not in _LLM_CACHE:
            _LLM_CACHE[cache_key] = PygmalionLLM(
                model_id=settings.default_model_id,
                device_map=device_map,
            )
        return _LLM_CACHE[cache_key]
    
    elif model_name == "gpt-oss-20b":
        repo_id = settings.gpt_oss_model_id
        if not repo_id:
            raise ValueError("gpt_oss_model_id is not configured.")
        device_map = _device_map(model_cfg)
        return load_gpt_oss_llm(
            model_id=repo_id,
            device_map=device_map,
        )

    raise ValueError(f"Not supported model: {model_name}")


__all__ = ["get_llm"]
