"""
GPT-OSS LLM adapter using vLLM for high-performance inference.

vLLM provides optimized inference with PagedAttention, continuous batching,
and efficient memory management for faster throughput compared to vanilla transformers.
"""
from __future__ import annotations
from functools import lru_cache
from typing import Dict, List, Sequence

from vllm import LLM, SamplingParams
from src.config.config import settings

_DEFAULT_STOP_STRINGS: List[str] = ["\nUser:", "\nUSER:", "\n사용자:"]


class GPTOssLLM:
    def __init__(
        self,
        model_id: str = "openai/gpt-oss-20b",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: int = 1024,
        trust_remote_code: bool = True,
    ):
        """
        Args:
            model_id: HuggingFace repo id for GPT-OSS variant.
            tensor_parallel_size: Number of GPUs for tensor parallelism.
            gpu_memory_utilization: Fraction of GPU memory to use (0.0-1.0).
            max_model_len: Maximum sequence length (None = auto-detect from model config).
            trust_remote_code: Whether to allow custom modeling code from the repo.
        """
        model_id = model_id or settings.gpt_oss_model_id
        self.model_id = model_id

        # vLLM automatically handles device placement and optimization
        self.llm = LLM(
            model=model_id,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            trust_remote_code=trust_remote_code,
            dtype="auto",  # vLLM auto-selects best dtype (bfloat16/float16)
        )

    def _trim(self, generated: str, stop_sequences: Sequence[str]) -> tuple[str, str]:
        cut_idx = len(generated)
        reason = "eos"
        for stop in stop_sequences:
            idx = generated.find(stop)
            if idx != -1 and idx < cut_idx:
                cut_idx = idx
                reason = "stop_string"
        return generated[:cut_idx].strip(), reason

    def generate(self, prompt: str, **gen) -> Dict:
        max_new_tokens = int(gen.get("max_new_tokens", 1024))
        temperature = float(gen.get("temperature", 0.8))
        top_p = float(gen.get("top_p", 0.95))
        repetition_penalty = float(gen.get("repetition_penalty", 1.05))
        stop = gen.get("stop") or []
        stop_sequences = _DEFAULT_STOP_STRINGS + list(stop)

        # vLLM SamplingParams
        sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            stop=stop_sequences,
        )

        # Generate (vLLM handles batching internally)
        outputs = self.llm.generate([prompt], sampling_params)
        output = outputs[0]
        
        # Extract generated text
        generated = output.outputs[0].text
        reply = generated.strip() if generated.strip() else "(…생각 중…)"
        
        # vLLM provides token counts
        prompt_tokens = len(output.prompt_token_ids)
        completion_tokens = len(output.outputs[0].token_ids)
        finish_reason = output.outputs[0].finish_reason  # 'stop', 'length', etc.

        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "stop_reason": finish_reason,
        }
        return {
            "reply": reply,
            "usage": usage,
            "raw": generated,
        }


@lru_cache(maxsize=2)
def load_gpt_oss_llm(
    model_id: str | None = None,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    max_model_len: int | None = None,
    trust_remote_code: bool = True,
) -> GPTOssLLM:
    """
    Cached factory so orchestrator can re-use heavyweight GPT-OSS vLLM instance.
    
    Args:
        model_id: HuggingFace repo id.
        tensor_parallel_size: Number of GPUs for tensor parallelism (default: 1).
        gpu_memory_utilization: GPU memory fraction (0.0-1.0, default: 0.9).
        max_model_len: Max sequence length (None = auto from model config).
        trust_remote_code: Allow custom modeling code.
    """
    return GPTOssLLM(
        model_id=model_id,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        trust_remote_code=trust_remote_code,
    )


__all__ = ["GPTOssLLM", "load_gpt_oss_llm"]
