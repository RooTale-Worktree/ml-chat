"""
EEVE-Korean-10.8B LLM adapter using vLLM for high-performance inference.
"""
from __future__ import annotations
from functools import lru_cache
from typing import Dict, List, Sequence

from vllm import LLM, SamplingParams
# from src.config.config import settings

# ❗ [변경] EEVE(ChatML) 템플릿은 <|im_end|>로 답변을 종료합니다.
_DEFAULT_STOP_STRINGS: List[str] = ["<|im_end|>", "<|im_start|>"]


class EeveeLLM:
    def __init__(
        self,
        model_id: str = "yanolja/YanoljaNEXT-EEVE-Instruct-10.8B",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: int | None = None,
        max_num_seqs: int | None = None,
        trust_remote_code: bool = True,
        dtype: str = "bfloat16",
    ):
        """
        Args:
            model_id: HuggingFace repo id for EEVE model.
            tensor_parallel_size: Number of GPUs for tensor parallelism.
            gpu_memory_utilization: Fraction of GPU memory to use (0.0-1.0).
            max_model_len: Maximum sequence length (None = auto-detect).
            max_num_seqs: Maximum number of sequences to process in parallel.
            trust_remote_code: Whether to allow custom modeling code.
            dtype: Data type for model weights (e.g., "auto", "float16", "bfloat16").
        """
        self.model_id = model_id

        self.llm = LLM(
            model=model_id,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            trust_remote_code=trust_remote_code,
            dtype=dtype,
        )

    def generate(self, prompt: str, **gen) -> Dict:
        """
        Generates a response from a pre-formatted prompt string.
        (This class expects the prompt to be *already* built)
        """
        max_new_tokens = int(gen.get("max_new_tokens", 1024))
        temperature = float(gen.get("temperature", 0.7))
        top_p = float(gen.get("top_p", 0.9))
        repetition_penalty = float(gen.get("repetition_penalty", 1.0))
        stop_sequences = _DEFAULT_STOP_STRINGS

        sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            stop=stop_sequences,
        )

        outputs = self.llm.generate([prompt], sampling_params)
        output = outputs[0]
        
        generated = output.outputs[0].text
        reply = generated.strip() if generated.strip() else "(…생각 중…)"
        
        prompt_tokens = len(output.prompt_token_ids)
        completion_tokens = len(output.outputs[0].token_ids)
        finish_reason = output.outputs[0].finish_reason

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
def load_eeve_llm(
    model_id: str | None = "yanolja/YanoljaNEXT-EEVE-Instruct-10.8B",
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    max_model_len: int | None = None,
    max_num_seqs: int | None = None,
    trust_remote_code: bool = True,
    dtype: str = "bfloat16",
) -> EeveeLLM:
    """
    Cached factory to re-use the heavyweight EEVE vLLM instance.
    """
    return EeveeLLM(
        model_id=model_id,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        trust_remote_code=trust_remote_code,
        dtype=dtype,
    )


__all__ = ["EeveeLLM", "load_eeve_llm"]