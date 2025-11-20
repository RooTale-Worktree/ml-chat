"""
High-level orchestration:
Combines RAG retrieval, Chat history, Persona and makes LLM calls to generate chat responses.
"""
from __future__ import annotations
import time
from typing import Dict, Optional, Any

from src.schemas.response import ChatResponse, ResponseContent, ModelInfo, Usage

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

    """
    Since chat RAG is not used, skip query embedding step.
    # Compute query embedding (normalized) for RAG
    tmp_time = time.time()
    query_embedding = embed_text(message)
    timing["message_embed_ms"] = int((time.time() - tmp_time) * 1000)
    """

    """
    Chat Rag is not used, rather all chat context is used for prompt directly.
    # Prompt element2: Chatting RAG context
    chat_context = []
    if chat_history:
        tmp_time = time.time()
        chat_rag = retrieve_chat_context(
            chat_history=chat_history,
            chat_rag_config=chat_rag_config,
            query_embedding=query_embedding,
        )
        timing["chat_retr_ms"] = int((time.time() - tmp_time) * 1000)
        chat_context = chat_rag.context
    """

    # Prompt element3: Story RAG context
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
        "chat_context": chat_history,
        "story_context": story_context,
        "user_message": message,
        "reasoning_effort": gen.get("reasoning_effort", "medium"),
    }
    model_name = model_config.get("name", None)
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

    """
    Since chat RAG is not used, skip response embedding step.
    # Embed response content
    tmp_time = time.time()
    resp_embedding = None
    resp_embedding = embed_text(gen_result["reply"])
    timing["response_embed_ms"] = int((time.time() - tmp_time) * 1000)
    """
    response_content = ResponseContent(
        content=gen_result["reply"],
    )

    usage_dict = gen_result.get("usage", {})
    usage = Usage(
        prompt_tokens=usage_dict.get("prompt_tokens", 0),
        completion_tokens=usage_dict.get("completion_tokens", 0),
        total_tokens=usage_dict.get("prompt_tokens", 0) + usage_dict.get("completion_tokens", 0),
        finish_reason=gen_result.get("usage", {}).get("stop_reason", None)
    )

    model_repo_name = getattr(llm, "model_id", model_name or settings.default_model_id)
    model_info = ModelInfo(
        name=model_repo_name,
        context_length=model_config.get("context_length", None),
        dtype=model_config.get("dtype", None),
        embedding_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    # Calculate total timing
    timing["total_ms"] = int((time.time() - start_time) * 1000)

    # Assemble final response
    resp = ChatResponse(
        session_id=req.session_id or "",
        responded_as="character",
        response_contents=[response_content],
        usage=usage,
        model_info=model_info,
        timing=timing,
    )
    return resp.model_dump()

__all__ = ["handle_chat"]
