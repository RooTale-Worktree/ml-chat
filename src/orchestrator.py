"""
High-level orchestration: combines RAG + persona + LLM.
"""
from __future__ import annotations
from typing import Dict

from src.schemas.request import ChatRequest
from src.schemas.response import ChatResponse, Choice, RetrievalItem, ModelInfo, Usage
from src.schemas.rag import PromptBuildInput

from src.core.prompt_builder import build_prompt
from src.rag.chat_rag import retrieve_chat_context
from src.rag.story_rag import retrieve_story_context
from src.core.embedding import embed_text

from src.llm.pygmalion_llm import PygmalionLLM
from src.llm.mock_llm import MockLLM
from src.config.config import settings

_llm_singleton = None

def _get_llm():
    """
    Lazy-load singleton Pygmalion model.
    In the future we could branch based on settings.env or a flag to use mock.
    """
    global _llm_singleton
    if _llm_singleton is None:
        _llm_singleton = PygmalionLLM(settings.default_model_id)
    return _llm_singleton


def handle_chat(payload: Dict) -> Dict:
    # Parse request
    req = ChatRequest(**payload)

    # Persona
    persona = req.persona

    # RAG
    # Compute query embedding (normalized) and pass to chat RAG
    try:
        query_embedding = embed_text(req.message)
    except Exception:
        # If embedding library isn't available, continue without it
        query_embedding = None

    chat_rag = retrieve_chat_context(
        message=req.message,
        history=req.history,
        chat_rag_config=req.chat_rag_config,
        query_embedding=query_embedding,
    )
    story_rag = retrieve_story_context()

    # Build prompt
    # Normalize history to DialogueTurn for prompt builder
    norm_history = [h.to_dialogue_turn() for h in req.history]

    prompt_input = PromptBuildInput(
        persona=persona,
        chat_context=chat_rag.context,
        story_context=story_rag.context,
        history=norm_history,
        user_message=req.message
    )
    prompt_out = build_prompt(prompt_input)

    # LLM generate
    model_name = req.model.name if req.model else None
    if model_name == "mock_llm":
        llm = MockLLM()
    elif model_name == "pygmalion-6b":
        llm = _get_llm()
    else:
        raise ValueError(f"Not supported model: {model_name}")
    gen_result = llm.generate(prompt_out.prompt, **req.gen.model_dump())

    # Map retrievals to response schema
    retrieved: list[RetrievalItem] = []
    for idx, ch in enumerate(chat_rag.context):
        retrieved.append(RetrievalItem(
            source="history",
            id=ch.id,
            role=None,
            content=ch.text,
            score=ch.score or 0.0,
            rank=idx,
            meta={"label": "CHAT"}
        ))
    base_rank = len(retrieved)
    for jdx, ch in enumerate(story_rag.context):
        retrieved.append(RetrievalItem(
            source="story",
            id=ch.id,
            role=None,
            content=ch.text,
            score=ch.score or 0.0,
            rank=base_rank + jdx,
            meta={"label": "STORY"}
        ))

    usage_dict = gen_result.get("usage", {})
    usage = Usage(
        prompt_tokens=usage_dict.get("prompt_tokens", 0),
        completion_tokens=usage_dict.get("completion_tokens", 0),
        total_tokens=usage_dict.get("prompt_tokens", 0) + usage_dict.get("completion_tokens", 0)
    )

    model_info = ModelInfo(
        provider="local",
        name=settings.default_model_id,
        context_length=req.model.context_length,
        dtype=req.model.dtype,
        embedding_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    resp = ChatResponse(
        session_id=req.session_id or "",
        responded_as="character",
        responded_character_id=req.persona.character_id,
        responded_character_name=req.persona.character_name,
        choices=[Choice(role="character", content=gen_result["reply"])],
        usage=usage,
        retrieved=retrieved,
        model_info=model_info,
        meta={
            "prompt": prompt_out.prompt,
            "prompt_meta": prompt_out.meta,
        }
    )
    return resp.model_dump()

__all__ = ["handle_chat"]