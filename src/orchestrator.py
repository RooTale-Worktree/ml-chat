"""
High-level orchestration: combines RAG + persona + LLM.
"""
from __future__ import annotations
import time
from typing import Dict

from src.schemas.request import ChatRequest
from src.schemas.response import ChatResponse, ResponseContent, RetrievalItem, ModelInfo, Usage, Timing
from src.schemas.rag import PromptBuildInput

from src.prompt.get_prompt import get_prompt
from src.rag.chat_rag import retrieve_chat_context
from src.rag.story_rag import retrieve_story_context
from src.core.embedding import embed_text

from src.llm.get_llm import get_llm
from src.config.config import settings


def handle_chat(payload: Dict) -> Dict:

    start_time = time.time()

    # Parse request
    req = ChatRequest(**payload)

    # Compute query embedding (normalized) for RAG
    t_message_embed_start = time.time()
    try:
        query_embedding = embed_text(req.message)
    except Exception:
        # If embedding library isn't available, continue without it
        query_embedding = None
    message_embed_ms = int((time.time() - t_message_embed_start) * 1000)

    # Prompt element1: Persona
    persona = req.persona

    chat_history = req.chat_history or []
    story_events = req.story or []

    # Prompt element2: Chatting RAG context
    chat_context = []
    if chat_history:
        t_chat_retr_start = time.time()
        chat_rag = retrieve_chat_context(
            chat_history=chat_history,
            chat_rag_config=req.chat_rag_config,
            query_embedding=query_embedding,
        )
        chat_retr_ms = int((time.time() - t_chat_retr_start) * 1000)
        chat_context = chat_rag.context
    else:
        chat_retr_ms = 0

    # Prompt element3: Story RAG context
    story_context = []
    if story_events:
        t_story_retr_start = time.time()
        story_rag = retrieve_story_context(
            story=story_events,
            user_query=req.message
        )
        story_retr_ms = int((time.time() - t_story_retr_start) * 1000)
        story_context = story_rag.context
    else:
        story_retr_ms = 0

    # Prompt element4: Recent chat history
    norm_history = [h.to_dialogue_turn() for h in chat_history]

    # Build prompt
    prompt_input = PromptBuildInput(
        persona=persona,
        chat_context=chat_context,
        story_context=story_context,
        recent_chat=norm_history,
        user_message=req.message
    )
    model_name = req.model.name if req.model else None
    prompt_out = get_prompt(model_name, prompt_input)

    # LLM generate
    t_llm_load_start = time.time()
    llm = get_llm(model_name, req.model)
    llm_load_ms = int((time.time() - t_llm_load_start) * 1000)

    # Generate response
    t_gen_start = time.time()
    gen_result = llm.generate(prompt_out.prompt, **req.gen.model_dump())
    generate_ms = int((time.time() - t_gen_start) * 1000)

    # Optionally embed response content (not counted in embed_ms; reported separately in meta)
    t_resp_embed_start = time.time()
    resp_embedding = None
    try:
        resp_embedding = embed_text(gen_result["reply"])
    except Exception:
        resp_embedding = None
    response_embed_ms = int((time.time() - t_resp_embed_start) * 1000)

    response_content = ResponseContent(
        content=gen_result["reply"],
        embedding=resp_embedding,
        character_id=req.persona.character_id,
        character_name=req.persona.character_name
    )

    # Map retrievals to response schema
    retrieved: list[RetrievalItem] = []
    for idx, ch in enumerate(chat_context):
        retrieved.append(RetrievalItem(
            source="chat_history",
            id=ch.id,
            role=None,
            content=ch.text,
            score=ch.score or 0.0,
            rank=idx,
            meta={"label": "CHAT"}
        ))
    base_rank = len(retrieved)
    for jdx, ch in enumerate(story_context):
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

    model_repo_name = getattr(llm, "model_id", model_name or settings.default_model_id)
    model_info = ModelInfo(
        provider="local",
        name=model_repo_name,
        context_length=req.model.context_length,
        dtype=req.model.dtype,
        embedding_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    # Calculate total timing and assemble timing informations
    total_ms = int((time.time() - start_time) * 1000)
    timing = Timing(
        total_ms=total_ms,
        message_embed_ms=message_embed_ms,
        response_embed_ms=response_embed_ms,
        rag_retr_ms={
            "chat_retr_ms": chat_retr_ms,
            "story_retr_ms": story_retr_ms
        },
        llm_load_ms=llm_load_ms,
        generate_ms=generate_ms,
    )

    # Assemble final response
    resp = ChatResponse(
        session_id=req.session_id or "",
        responded_as="character",
        response_contents=response_content,
        usage=usage,
        retrieved=retrieved,
        model_info=model_info,
        timing=timing,
        meta={
            "prompt": prompt_out.prompt,
            "prompt_meta": prompt_out.meta,
        }
    )
    return resp.model_dump()

__all__ = ["handle_chat"]
