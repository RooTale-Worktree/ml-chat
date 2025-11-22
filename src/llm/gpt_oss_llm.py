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

_DEFAULT_STOP_STRINGS: List[str] = ["USER:", "\nUSER:"]


class GPTOssLLM:
    def __init__(
        self,
        model_id: str = "openai/gpt-oss-20b",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: int = 1024,
        max_num_seqs: int | None = None,
        trust_remote_code: bool = True,
        dtype: str = "auto",
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
            max_num_seqs=max_num_seqs,
            trust_remote_code=trust_remote_code,
            dtype=dtype,
        )
    
    def _parse_generated_text(self, text: str):
        """
        Parse generated text to separate CoT and final reply.
        """
        cot_content = ""
        reply_content = text

        # 1. Reply 마커로 분리 시도
        if "assistantfinal=" in text:
            parts = text.split("assistantfinal=", 1)
            cot_content = parts[0]
            reply_content = parts[1].strip()
        elif "assistantfinal" in text:
            parts = text.split("assistantfinal", 1)
            cot_content = parts[0]
            reply_content = parts[1].strip()            
            # 2. CoT 마커 제거 (앞부분에 붙어있다면)
        if cot_content.startswith("analysis"):
            cot_content = cot_content[len("analysis"):].strip()
        else:
            cot_content = cot_content.strip()
        
        return cot_content, reply_content

    def generate(self, prompt: str, **gen) -> Dict:
        max_new_tokens = int(gen.get("max_new_tokens", 1024))
        temperature = float(gen.get("temperature", 0.8))
        top_p = float(gen.get("top_p", 0.95))
        repetition_penalty = float(gen.get("repetition_penalty", 1.05))
        stop_sequences = gen.get("stop", _DEFAULT_STOP_STRINGS)

        # vLLM SamplingParams
        sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            stop=stop_sequences,
        )

        # Generate (vLLM handles batching internally)
        outputs = self.llm.generate(prompt, sampling_params)
        output = outputs[0]
        
        # Extract generated text
        generated_full_text = output.outputs[0].text
        cot, reply = self._parse_generated_text(generated_full_text)

        print(f"[Handler] CoT: {cot}")
        print(f"[Handler] Reply: {reply}")
        
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
            "cot": cot,
        }


@lru_cache(maxsize=2)
def load_gpt_oss_llm(
    model_id: str | None = None,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    max_model_len: int | None = None,
    max_num_seqs: int | None = None,
    trust_remote_code: bool = True,
    dtype: str | None = None,
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
        max_num_seqs=max_num_seqs,
        trust_remote_code=trust_remote_code,
        dtype=dtype,
    )


__all__ = ["GPTOssLLM", "load_gpt_oss_llm"]
