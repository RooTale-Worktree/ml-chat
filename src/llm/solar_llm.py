from __future__ import annotations
from functools import lru_cache
from typing import Dict, List, Sequence

from vllm import LLM, SamplingParams
from src.config.config import settings


_DEFAULT_STOP_STRINGS: List[str] = ["[INST]", "</s>"]


class SolarLLM:
    def __init__(
        self,
        model_id: str = "upstage/SOLAR-10.7B-Instruct-v1.0",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: int | None = None,
        max_num_seqs: int | None = None,
        trust_remote_code: bool = True,
        dtype: str = "bfloat16",
        meta: Dict | None = None,
    ):
        """
        Args:
            model_id: HuggingFace repo id for SOLAR model.
            tensor_parallel_size: Number of GPUs for tensor parallelism.
            gpu_memory_utilization: Fraction of GPU memory to use (0.0-1.0).
            max_model_len: Maximum sequence length (None = auto-detect).
            trust_remote_code: Whether to allow custom modeling code.
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
            **(meta or {})
        )
        
        self.hf_tokenizer = self.llm.llm_engine.tokenizer

    def generate(self, messages: List[Dict[str, str]], **gen) -> Dict:
        """
        Generates a response using the SOLAR chat template.
        
        Args:
            messages: A list of message dictionaries, e.g.,
                      [{"role": "system", "content": "..."},
                       {"role": "user", "content": "..."}]
            **gen: Generation parameters (max_new_tokens, temperature, etc.)
        """
        max_new_tokens = int(gen.get("max_new_tokens", 1024))
        temperature = float(gen.get("temperature", 0.7))
        top_p = float(gen.get("top_p", 0.9))
        repetition_penalty = float(gen.get("repetition_penalty", 1.0))
        stop_sequences = _DEFAULT_STOP_STRINGS

        # vLLM SamplingParams
        sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            stop=stop_sequences,
        )

        try:
            prompt_str = self.hf_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except Exception as e:
            raise ValueError(f"Error applying chat template: {e}")

        outputs = self.llm.generate([prompt_str], sampling_params)
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
def load_solar_llm(
    model_id: str | None = "upstage/SOLAR-10.7B-Instruct-v1.0",
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    max_model_len: int | None = None,
    max_num_seqs: int | None = None,
    trust_remote_code: bool = True,
    dtype: str = "bfloat16",
    meta: Dict | None = None,
) -> SolarLLM:
    """
    Cached factory to re-use the heavyweight SOLAR vLLM instance.
    """
    return SolarLLM(
        model_id=model_id,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        trust_remote_code=trust_remote_code,
        dtype=dtype,
        meta=meta,
    )


__all__ = ["SolarLLM", "load_solar_llm"]