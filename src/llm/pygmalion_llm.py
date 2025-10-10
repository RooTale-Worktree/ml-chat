"""
Pygmalion LLM adapter using HuggingFace Transformers.

This wraps loading + generation so orchestrator can switch between mock and real model.

NOTES:
- For serverless cold start, consider lazy loading and optionally quantization.
- Stop string trimming kept simple; can be extended.
"""
from __future__ import annotations
from typing import Dict, Optional, List
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_STOP_STRINGS: List[str] = ["\nUser:", "\nUSER:", "\n사용자:", "\n"]

class PygmalionLLM:
    def __init__(self, model_id: str, device_map: str | None = "auto"):
        if torch.cuda.is_available():
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            dtype = torch.float32
            device_map = None
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map=device_map,
        )
        self.model.eval()

    def _trim(self, prompt: str, generated: str):
        cut_idx = len(generated)
        reason = "eos"
        for s in _STOP_STRINGS:
            idx = generated.find(s)
            if idx != -1 and idx < cut_idx:
                cut_idx = idx
                reason = "stop_string"
        return generated[:cut_idx].strip(), reason

    def generate(self, prompt: str, **gen) -> Dict:
        max_new_tokens = int(gen.get("max_new_tokens", 256))
        temperature = float(gen.get("temperature", 0.8))
        top_p = float(gen.get("top_p", 0.9))
        repetition_penalty = float(gen.get("repetition_penalty", 1.1))

        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids
        if torch.cuda.is_available():
            input_ids = input_ids.to(self.model.device)

        with torch.no_grad():
            out_ids = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        full = self.tokenizer.decode(out_ids[0], skip_special_tokens=True)
        generated = full[len(prompt):]
        reply, stop_reason = self._trim(prompt, generated)

        return {
            "reply": reply if reply else "(…생각 중…) ",
            "usage": {
                # crude token estimates
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": len(reply.split()),
                "stop_reason": stop_reason,
            },
            "raw": generated,
        }

__all__ = ["PygmalionLLM"]
