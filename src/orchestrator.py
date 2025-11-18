"""
High-level orchestration: combines RAG + persona + LLM.
"""
from __future__ import annotations
import time
from typing import Dict, Optional, Any

from src.schemas.request import ChatRequest
from src.schemas.response import ChatResponse, ResponseContent, ModelInfo, Usage
from src.schemas.rag import PromptBuildInput

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
        llm_instance: 사전 로딩된 LLM 인스턴스 (RunPod serverless 최적화용)
                     None이면 동적으로 로딩 (로컬 테스트용)
    
    Returns:
        ChatResponse dict
    """
    timing = {}
    start_time = time.time()

    # Parse request
    req = ChatRequest(**payload)

    # Compute query embedding (normalized) for RAG
    tmp_time = time.time()
    query_embedding = embed_text(req.message)
    timing["message_embed_ms"] = int((time.time() - tmp_time) * 1000)

    # Prompt element1: Persona
    persona = req.persona

    chat_history = req.chat_history or []
    story_events = req.story or []

    # Prompt element2: Chatting RAG context
    chat_context = []
    if chat_history:
        tmp_time = time.time()
        chat_rag = retrieve_chat_context(
            chat_history=chat_history,
            chat_rag_config=req.chat_rag_config,
            query_embedding=query_embedding,
        )
        timing["chat_retr_ms"] = int((time.time() - tmp_time) * 1000)
        chat_context = chat_rag.context

    # Prompt element3: Story RAG context
    story_context = []
    
    if req.story_title:
        tmp_time = time.time()
        story_rag = retrieve_story_context(
            story_title=req.story_title,  
            user_query=req.message
        )
        timing["story_retr_ms"] = int((time.time() - tmp_time) * 1000)
        story_context = story_rag.context

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
    model_name = req.model.get("name", None)
    prompt_out = get_prompt(model_name, prompt_input)

    # LLM generate - 사전 로딩된 인스턴스 사용 또는 동적 로딩
    if llm_instance is None:
        # 로컬 테스트: 동적 로딩
        tmp_time = time.time()
        llm = get_llm(model_name, req.model)
        timing["llm_load_ms"] = int((time.time() - tmp_time) * 1000)
    else:
        # RunPod serverless: 사전 로딩된 인스턴스 재사용
        llm = llm_instance
        timing["llm_load_ms"] = 0  # 이미 로딩됨

    # Generate response
    tmp_time = time.time()
    gen_result = llm.generate(prompt_out.prompt, **req.gen.model_dump())
    timing["generate_ms"] = int((time.time() - tmp_time) * 1000)

    # Embed response content
    tmp_time = time.time()
    resp_embedding = None
    resp_embedding = embed_text(gen_result["reply"])
    timing["response_embed_ms"] = int((time.time() - tmp_time) * 1000)

    response_content = ResponseContent(
        content=gen_result["reply"],
        embedding=resp_embedding,
        character_id=req.persona.character_id,
        character_name=req.persona.character_name,
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
        context_length=req.model.get("context_length", None),
        dtype=req.model.get("dtype", None),
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
