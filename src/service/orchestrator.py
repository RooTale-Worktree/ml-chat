from __future__ import annotations
"""High-level orchestration: combines RAG + persona + LLM."""
from typing import Dict
from ..core.schemas import (
    ChatRequest, ChatResponse, PromptBuildInput
)
from ..core.persona import load_persona
from ..rag.chat_rag import retrieve_chat_context
from ..rag.story_rag import retrieve_story_context
from ..core.prompt_builder import build_prompt
from ..llm.pygmalion_llm import PygmalionLLM
from ..config.config import settings

_llm_singleton = None

def _get_llm():
    """Lazy-load singleton real Pygmalion model.
    In the future we could branch based on settings.env or a flag to use mock.
    """
    global _llm_singleton
    if _llm_singleton is None:
        _llm_singleton = PygmalionLLM(settings.default_model_id)
    return _llm_singleton


def handle_chat(payload: Dict) -> Dict:
    # Parse request
    req = ChatRequest(**payload)

    # Persona already provided in request; if not, could fallback load_persona()
    persona = req.persona

    # RAG steps
    chat_rag = retrieve_chat_context(req.history)
    story_rag = retrieve_story_context()

    # Build prompt
    prompt_input = PromptBuildInput(
        persona=persona,
        chat_context=chat_rag.context,
        story_context=story_rag.context,
        history=req.history,
        user_message=req.message
    )
    prompt_out = build_prompt(prompt_input)

    # LLM generate (real pygmalion)
    llm = _get_llm()
    gen_result = llm.generate(prompt_out.prompt, **req.gen)

    resp = ChatResponse(
        reply=gen_result["reply"],
        prompt=prompt_out.prompt,
        chat_context=chat_rag.context,
        story_context=story_rag.context,
        meta={
            "model_id": settings.default_model_id,
            "usage": gen_result.get("usage", {}),
            "prompt_meta": prompt_out.meta
        }
    )
    return resp.model_dump()

__all__ = ["handle_chat"]
