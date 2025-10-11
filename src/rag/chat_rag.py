"""
Chat (recent dialogue) retrieval stub.
For now, returns last N user/character turns as pseudo-context.
Later can be replaced by semantic search over dialogue memory store.
"""
from __future__ import annotations
from typing import List, Union

from src.schemas.request import Message, ChatRAGConfig
from src.schemas.rag import DialogueTurn, RAGChunk, ChatRAGResult
from src.config.config import settings

_DEF_SOURCE = "chat_history"

def retrieve_chat_context(message: str,
                          history: List[Message],
                          chat_rag_config: ChatRAGConfig,
                          query_embedding: List[float] | None = None) -> ChatRAGResult:
    # Normalize to DialogueTurn
    norm: List[DialogueTurn] = []
    for h in history:
        if isinstance(h, Message):
            norm.append(h.to_dialogue_turn())
        else:
            norm.append(h)
    # TODO: If query_embedding is provided, perform semantic retrieval against message_embeddings
    # For now, keep the simple recency-based fallback
    turns = norm[-settings.chat_k:]
    chunks: List[RAGChunk] = []
    for i, t in enumerate(turns):
        chunks.append(RAGChunk(
            id=f"chat-{i}",
            source=_DEF_SOURCE,
            text=f"{t.role}: {t.content}",
            score=None
        ))
    return ChatRAGResult(context=chunks)

__all__ = ["retrieve_chat_context"]
