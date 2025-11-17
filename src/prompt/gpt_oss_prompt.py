"""
Prompt builder that renders persona + RAG context using Harmony format for gpt-oss.
"""
from __future__ import annotations
from datetime import datetime, timezone
from typing import List

from src.schemas.request import DialogueTurn
from src.schemas.rag import PromptBuildInput, PromptBuildOutput, RAGChunk

prompt_template = """
당신은 '{character_name}'입니다. '{character_name}'의 persona를 연기해야 합니다.
- 이름: {character_name}
- 성격: {persona}
- 설정: {scenario}
- 대화 스타일: {speaking_style}
- 제약 조건: {constraints}

---
[예시 대화]
{example_dialogue}
---
[참고 맥락]
{rag_context}
---
[대화 기록]
{chat_history}
USER: {user_input}
ASSISTANT:assistantfinal"""


def build_prompt(data: PromptBuildInput) -> PromptBuildOutput:

    persona = data.persona
    
    # 1. example dialogue
    example_lines = []
    for turn in persona.example_dialogue:
        role = persona.character_name if turn.role in ["assistant", "character"] else "User"
        example_lines.append(f"{role}: {turn.content}")
    example_dialogue = "\n".join(example_lines) if example_lines else "(예시 없음)"
    
    # 2. RAG context
    rag_lines = []
    if data.chat_context:
        rag_lines.append("[대화 기억]")
        for chunk in data.chat_context:
            rag_lines.append(f"- {chunk.text}")
    if data.story_context:
        rag_lines.append("[스토리 정보]")
        for chunk in data.story_context:
            rag_lines.append(f"- {chunk.text}")
    rag_context = "\n".join(rag_lines) if rag_lines else "(참고 맥락 없음)"
    
    # 3. Recent chat history
    chat_lines = []
    for turn in data.recent_chat:
        role = persona.character_name if turn.role in ["assistant", "character"] else "User"
        chat_lines.append(f"{role}: {turn.content}")
    chat_history = "\n".join(chat_lines) if chat_lines else "(대화 기록 없음)"
    
    # 4. Constraints
    constraints = "; ".join(persona.constraints) if persona.constraints else "없음"
    
    # 5. Fill template with values
    prompt = prompt_template.format(
        character_name=persona.character_name,
        persona=persona.persona,
        scenario=persona.scenario,
        speaking_style=persona.speaking_style,
        constraints=constraints,
        example_dialogue=example_dialogue,
        rag_context=rag_context,
        chat_history=chat_history,
        user_input=data.user_message
    )
    
    # 6. Prepare metadata
    meta = {
        "chat_ctx_count": len(data.chat_context),
        "story_ctx_count": len(data.story_context),
        "recent_chat_count": len(data.recent_chat),
    }
    
    return PromptBuildOutput(prompt=prompt, meta=meta)


__all__ = ["build_prompt"]
