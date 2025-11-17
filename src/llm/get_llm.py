"""
Central place to load LLM instances depending on requested model.
"""
from __future__ import annotations

from src.config.config import settings
from src.llm.gpt_oss_llm import load_gpt_oss_llm


def get_llm(model_name: str | None, model_cfg: dict | None = None):
    """
    Return an LLM instance for the requested model.
    
    Note: In RunPod serverless environment, each request may get a new container,
    so we rely on vLLM's internal optimizations rather than application-level caching.
    The load_gpt_oss_llm function still uses @lru_cache for same-process reuse.
    """
    if model_name == "gpt-oss-20b":
        repo_id = settings.gpt_oss_model_id
        tensor_parallel = model_cfg.get("tensor_parallel_size", 1)
        gpu_mem_util = model_cfg.get("gpu_memory_utilization", 0.9)
        max_model_len = model_cfg.get("max_model_len", 1024)
        trust_remote_code = model_cfg.get("trust_remote_code", True)
        
        return load_gpt_oss_llm(
            model_id=repo_id,
            tensor_parallel_size=tensor_parallel,
            gpu_memory_utilization=gpu_mem_util,
            max_model_len=max_model_len,
            trust_remote_code=trust_remote_code,
        )
    else:
        raise ValueError(f"Not supported model: {model_name}")


__all__ = ["get_llm"]
