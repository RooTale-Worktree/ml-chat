"""
Prompt builder that renders persona + RAG context for EEVE models
using tokenizer.apply_chat_template (ChatML format).
"""
from __future__ import annotations
from typing import List, Dict
from functools import lru_cache

from transformers import AutoTokenizer

from src.schemas.request import DialogueTurn
from src.schemas.rag import PromptBuildInput, PromptBuildOutput, RAGChunk


@lru_cache(maxsize=1)
def _get_eeve_tokenizer():
    """
    Caches the EEVE tokenizer to avoid reloading it on every call.
    """
    return AutoTokenizer.from_pretrained(
        "tuni/EEVE-Korean-10.8B-v1.0",
        trust_remote_code=True
    )


def build_prompt(prompt_input: PromptBuildInput) -> PromptBuildOutput:
    """
    Builds an EEVE-compatible prompt string using apply_chat_template.
    (This logic is identical to build_prompt_solar as both benefit
     from structured chat templates).
    """
    
    tokenizer = _get_eeve_tokenizer()
    persona = prompt_input.persona
    
    rag_lines = []
    if prompt_input.chat_context:
        rag_lines.append("[대화 기억]")
        for chunk in prompt_input.chat_context:
            rag_lines.append(f"- {chunk.text}")
    if prompt_input.story_context:
        rag_lines.append("[스토리 정보]")
        for chunk in prompt_input.story_context:
            rag_lines.append(f"- {chunk.text}")
    rag_context = "\n".join(rag_lines) if rag_lines else "(참고 맥락 없음)"

    constraints = "; ".join(persona.constraints) if persona.constraints else "없음"
    
    system_content = f"""[임무]
당신은 '{persona.character_name}'입니다. 사용자에게 '{persona.character_name}'로서 응답해야 합니다.

[페르소나: '{persona.character_name}']
- 이름: {persona.character_name}
- 성격: {persona.persona}
- 설정: {persona.scenario}
- 대화 스타일: {persona.speaking_style}
- 제약 조건: {constraints}

[참고 맥락]
{rag_context}
"""
    
    messages: List[Dict[str, str]] = []
    messages.append({"role": "system", "content": system_content})

    for turn in persona.example_dialogue:
        role = "assistant" if turn.role in ["assistant", "character"] else "user"
        messages.append({"role": role, "content": turn.content})

    for turn in prompt_input.recent_chat:
        role = "assistant" if turn.role in ["assistant", "character"] else "user"
        messages.append({"role": role, "content": turn.content})

    messages.append({"role": "user", "content": prompt_input.user_message})

    try:
        prompt_str = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    except Exception as e:
        raise ValueError(f"Error applying chat template: {e}")

    meta = {
        "chat_ctx_count": len(prompt_input.chat_context),
        "story_ctx_count": len(prompt_input.story_context),
        "recent_chat_count": len(prompt_input.recent_chat),
        "example_chat_count": len(persona.example_dialogue),
    }
    
    return PromptBuildOutput(prompt=prompt_str, meta=meta)


__all__ = ["build_prompt"]