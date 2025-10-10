"""
Mock LLM adapter for local testing without heavy model downloads.
"""
from __future__ import annotations
from dataclasses import dataclass
import random
from typing import Dict

@dataclass
class MockLLMConfig:
    seed: int = 42

class MockLLM:
    def __init__(self, config: MockLLMConfig | None = None):
        self.config = config or MockLLMConfig()
        random.seed(self.config.seed)

    def generate(self, prompt: str, **gen) -> Dict:
        # naive generation stub
        endings = [
            "... 흥미롭군요.",
            " 그게 네가 원하는 거야?", 
            " 내가 그렇게 말한 적은 없는데.",
            " 더 자세히 말해줄래?",
        ]
        reply = "응답: " + random.choice(endings)
        return {
            "reply": reply,
            "raw": reply,
            "usage": {"prompt_tokens": len(prompt.split()), "completion_tokens": 6}
        }

__all__ = ["MockLLM", "MockLLMConfig"]
