"""
Prompt builder tailored for GPT-OSS style chat completion.
"""
from __future__ import annotations
from typing import List

from src.schemas.request import DialogueTurn
from src.schemas.rag import PromptBuildInput, PromptBuildOutput, RAGChunk

SYSTEM_TEMPLATE = (
    "### Instruction\n"
    "You are roleplaying as {name}. Answer in the tone described by the persona while"
    " respecting every constraint. Avoid breaking character unless explicitly asked.\n"
    "### Persona\n"
    "- Description: {persona}\n"
    "- Scenario: {scenario}\n"
    "- Speaking Style: {style}\n"
    "- Constraints: {constraints}\n"
)

CONTEXT_TEMPLATE = "### {label} Context\n{body}\n"
EXAMPLE_TEMPLATE = "### Few-shot Dialogue\n{body}\n"
CONVERSATION_TEMPLATE = (
    "### Conversation\n"
    "{history}\n"
    "User: {user_message}\n"
    "{name}:"
)


def _fmt_examples(persona_name: str, turns: List[DialogueTurn]) -> str:
    if not turns:
        return ""
    lines = []
    for t in turns:
        role = persona_name if t.role == "character" else "User"
        lines.append(f"{role}: {t.content}")
    return EXAMPLE_TEMPLATE.format(body="\n".join(lines))


def _fmt_chunks(chunks: List[RAGChunk]) -> str:
    return "\n---\n".join(ch.text for ch in chunks if ch.text)


def _fmt_history(char_name: str, turns: List[DialogueTurn]) -> str:
    if not turns:
        return ""
    lines = []
    for t in turns:
        role = "User" if t.role == "user" else char_name
        lines.append(f"{role}: {t.content}")
    return "\n".join(lines)


def build_prompt(data: PromptBuildInput) -> PromptBuildOutput:
    persona = data.persona
    constraints = "; ".join(persona.constraints) if persona.constraints else "None"
    system_block = SYSTEM_TEMPLATE.format(
        name=persona.character_name,
        persona=persona.persona,
        scenario=persona.scenario,
        style=persona.speaking_style,
        constraints=constraints,
    )
    examples_block = _fmt_examples(persona.character_name, persona.example_dialogue)
    chat_block = (
        CONTEXT_TEMPLATE.format(label="Chat", body=_fmt_chunks(data.chat_context))
        if data.chat_context else ""
    )
    story_block = (
        CONTEXT_TEMPLATE.format(label="Story", body=_fmt_chunks(data.story_context))
        if data.story_context else ""
    )
    convo_block = CONVERSATION_TEMPLATE.format(
        history=_fmt_history(persona.character_name, data.recent_chat),
        user_message=data.user_message,
        name=persona.character_name,
    )

    prompt = system_block + examples_block + chat_block + story_block + convo_block
    meta = {
        "chat_ctx_count": len(data.chat_context),
        "story_ctx_count": len(data.story_context),
    }
    return PromptBuildOutput(prompt=prompt, meta=meta)


__all__ = ["build_prompt"]
