"""
High-level orchestration:
Combines RAG retrieval, Chat history, Persona and makes LLM calls to generate chat responses.
"""
from __future__ import annotations
import time
from typing import Dict, Optional, Any

from src.schemas.response import ChatResponse, ResponseContent, Timing, Usage

from src.llm.get_llm import get_llm
from src.prompt.get_prompt import get_prompt
from src.rag.chat_rag import retrieve_chat_context
from src.rag.story_rag import retrieve_story_context
from src.core.embedding import embed_text
from src.config.config import settings


def handle_chat(payload: Dict, llm_instance: Optional[Any] = None) -> Dict:
    """
    Handle a single chat turn.    
    Args:
        payload: ChatRequest dict
        llm_instance: Pre-loaded LLM instance. If None, load dynamically.
    Returns:
        ChatResponse: dict
    """
    timing = {}
    start_time = time.time()

    # Parse payload
    message = payload.get("message", "")
    user_name = payload.get("user_name", "User")
    persona = payload.get("persona", None)
    chat_history = payload.get("chat_history", [])
    chat_rag_config = payload.get("chat_rag_config", {})
    story_title = payload.get("story_title", None)
    model_config = payload.get("model_config", {})
    gen = payload.get("gen", {})
    meta = payload.get("meta", {})

    model_name = model_config.get("model_name", None)

    # Compute query embedding (normalized) for RAG
    # if chat_rag_config is provided
    tmp_time = time.time()
    if chat_rag_config:
        query_embedding = embed_text(message)
        timing["message_embed_ms"] = int((time.time() - tmp_time) * 1000)

    # Prompt element1: Use chat RAG if chat_rag_config is provided
    # else use full chat history
    chat_context = []
    if chat_history and chat_rag_config:
        tmp_time = time.time()
        chat_rag = retrieve_chat_context(
            chat_history=chat_history,
            chat_rag_config=chat_rag_config,
            query_embedding=query_embedding,
        )
        timing["chat_retr_ms"] = int((time.time() - tmp_time) * 1000)
        chat_context = chat_rag.context
    elif chat_history:
        chat_context = chat_history

    # Prompt element2: Story RAG context
    story_context = []    
    if story_title:
        tmp_time = time.time()
        story_context = retrieve_story_context(
            story_title=story_title,
            user_query=message
        )
        timing["story_retr_ms"] = int((time.time() - tmp_time) * 1000)

    # Build prompt
    prompt_input = {
        "persona": persona,
        "user_name": user_name,
        "chat_context": chat_context,
        "story_context": story_context,
        "user_message": message,
        "reasoning_effort": gen.get("reasoning_effort", "medium"),
    }
    prompt = get_prompt(model_name, prompt_input)

    # Generate LLM response
    if llm_instance is None:
        # Load LLM dynamically
        tmp_time = time.time()
        llm = get_llm(model_name, model_config)
        timing["llm_load_ms"] = int((time.time() - tmp_time) * 1000)
    else:
        # Use pre-loaded LLM instance
        llm = llm_instance
        timing["llm_load_ms"] = 0

    # Generate response
    tmp_time = time.time()
    gen_result = llm.generate(prompt, **gen)
    timing["generate_ms"] = int((time.time() - tmp_time) * 1000)

    # Embed response content if chat_rag_config is provided
    resp_embedding = None
    if chat_rag_config:
        tmp_time = time.time()
        resp_embedding = embed_text(gen_result["reply"])
        timing["response_embed_ms"] = int((time.time() - tmp_time) * 1000)

    response_content = ResponseContent(
        content=gen_result["reply"],
        embedding=resp_embedding
    )

    usage_dict = gen_result.get("usage", {})
    usage = Usage(
        prompt_tokens=usage_dict.get("prompt_tokens", 0),
        completion_tokens=usage_dict.get("completion_tokens", 0),
        total_tokens=usage_dict.get("prompt_tokens", 0) + usage_dict.get("completion_tokens", 0),
        finish_reason=gen_result.get("usage", {}).get("stop_reason", None)
    )

    # Calculate total timing
    timing["total_ms"] = int((time.time() - start_time) * 1000)
    timing_obj = Timing(**timing)

    # Assemble final response
    resp = ChatResponse(
        responded_as="character",
        response_contents=[response_content],
        usage=usage,
        timing=timing_obj,
    )
    return resp.model_dump()

__all__ = ["handle_chat"]
