"""
GPT-OSS LLM adapter loaded from HuggingFace.

This mirrors Pygmalion adapter but keeps the naming/theme for GPT-OSS
so orchestrator can switch between the two models seamlessly.
"""
from __future__ import annotations
from functools import lru_cache
from typing import Dict, List, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.config.config import settings

_DEFAULT_STOP_STRINGS: List[str] = ["\nUser:", "\nUSER:", "\n사용자:", "\n"]


class GPTOssLLM:
    def __init__(
        self,
        model_id: str | None = None,
        device_map: str | None = "auto",
        attn_implementation: str | None = None,
        trust_remote_code: bool = True,
    ):
        """
        Args:
            model_id: HuggingFace repo id for GPT-OSS variant.
            device_map: Passed to transformers for dispatch (defaults to "auto").
            attn_implementation: Optional attention backend hint (e.g., "flash_attention_2").
            trust_remote_code: Whether to allow custom modeling code from the repo.
        """
        model_id = model_id or settings.gpt_oss_model_id

        if torch.cuda.is_available():
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            dtype = torch.float32
            device_map = None

        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            use_fast=True,
            trust_remote_code=trust_remote_code,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs = {
            "dtype": dtype,
            "device_map": device_map,
            "trust_remote_code": trust_remote_code,
        }
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            **model_kwargs,
        )
        self.model.eval()

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
        max_new_tokens = int(gen.get("max_new_tokens", 256))
        temperature = float(gen.get("temperature", 0.8))
        top_p = float(gen.get("top_p", 0.95))
        repetition_penalty = float(gen.get("repetition_penalty", 1.05))
        stop = gen.get("stop") or []
        stop_sequences = _DEFAULT_STOP_STRINGS + list(stop)

        inputs = self.tokenizer(prompt, return_tensors="pt", padding=False)
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")
        if torch.cuda.is_available():
            device = self.model.device
            input_ids = input_ids.to(device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

        with torch.inference_mode():
            out_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        decoded = self.tokenizer.decode(out_ids[0], skip_special_tokens=True)
        generated = decoded[len(prompt):]
        reply, stop_reason = self._trim(generated, stop_sequences)
        reply = reply if reply else "(…생각 중…)"

        usage = {
            "prompt_tokens": len(prompt.split()),
            "completion_tokens": len(reply.split()),
            "stop_reason": stop_reason,
        }
        return {
            "reply": reply,
            "usage": usage,
            "raw": generated,
        }


@lru_cache(maxsize=2)
def load_gpt_oss_llm(
    model_id: str | None = None,
    device_map: str | None = "auto",
    attn_implementation: str | None = None,
    trust_remote_code: bool = True,
) -> GPTOssLLM:
    """
    Cached factory so orchestrator can re-use heavyweight GPT-OSS weights.
    """
    return GPTOssLLM(
        model_id=model_id,
        device_map=device_map,
        attn_implementation=attn_implementation,
        trust_remote_code=trust_remote_code,
    )


__all__ = ["GPTOssLLM", "load_gpt_oss_llm"]
