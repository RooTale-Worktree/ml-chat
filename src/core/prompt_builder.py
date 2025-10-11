"""
Prompt assembly combining persona + chat context + story context + history + user message.
"""
from __future__ import annotations
from typing import List

from src.schemas.request import DialogueTurn
from src.schemas.rag import PromptBuildInput, PromptBuildOutput, RAGChunk


SYSTEM_TEMPLATE = """<<SYSTEM>>\nYou are roleplaying as the character below. Stay in character.\nName: {name}\nPersona: {persona}\nScenario: {scenario}\nStyle: {style}\nConstraints: {constraints}\n<<END_SYSTEM>>\n"""

EXAMPLES_HEADER = "<<EXAMPLES>>\n{examples}\n<<END_EXAMPLES>>\n"
CONTEXT_HEADER = "<<CONTEXT:{label}>>\n{body}\n<<END_CONTEXT>>\n"
CONVERSATION_HEADER = "<<CONVERSATION>>\n{history}\nUser: {user}\n{name}:"

def _format_examples(persona) -> str:
    if not persona.example_dialogue:
        return ""
    lines = []
    for t in persona.example_dialogue:
        role = persona.name if str(t.role).lower() == "character" else "User"
        lines.append(f"{role}: {t.content}")
    return EXAMPLES_HEADER.format(examples="\n".join(lines))


def _fmt_chunks(chunks: List[RAGChunk]) -> str:
    return "\n---\n".join(ch.text for ch in chunks)


def _fmt_history(turns: List[DialogueTurn], char_name: str) -> str:
    lines = []
    for t in turns:
        role = "User" if t.role == "user" else char_name
        lines.append(f"{role}: {t.content}")
    return "\n".join(lines)


def build_prompt(data: PromptBuildInput) -> PromptBuildOutput:
    p = data.persona
    system_block = SYSTEM_TEMPLATE.format(
        name=p.name,
        persona=p.persona,
        scenario=p.scenario,
        style=p.speaking_style,
        constraints="; ".join(p.constraints)
    )
    examples_block = _format_examples(p)
    chat_ctx_block = CONTEXT_HEADER.format(label="CHAT", body=_fmt_chunks(data.chat_context)) if data.chat_context else ""
    story_ctx_block = CONTEXT_HEADER.format(label="STORY", body=_fmt_chunks(data.story_context)) if data.story_context else ""
    hist_block = _fmt_history(data.history, p.name)

    prompt = (
        system_block
        + examples_block
        + chat_ctx_block
        + story_ctx_block
    + CONVERSATION_HEADER.format(history=hist_block, user=data.user_message, name=p.name)
    )

    return PromptBuildOutput(prompt=prompt, meta={
        "chat_ctx_count": len(data.chat_context),
        "story_ctx_count": len(data.story_context)
    })

__all__ = ["build_prompt"]
