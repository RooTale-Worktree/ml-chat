from __future__ import annotations
"""Chat (recent dialogue) retrieval stub.
For now, returns last N user/character turns as pseudo-context.
Later can be replaced by semantic search over dialogue memory store.
"""
from typing import List
from ..core.schemas import DialogueTurn, RAGChunk, ChatRAGResult
from ..config.config import settings

_DEF_SOURCE = "chat_history"

def retrieve_chat_context(history: List[DialogueTurn]) -> ChatRAGResult:
    turns = history[-settings.chat_k:]
    chunks: List[RAGChunk] = []
    for i, t in enumerate(turns):
        chunks.append(RAGChunk(
            id=f"chat-{i}",
            source=_DEF_SOURCE,
            text=f"{t.role}: {t.text}",
            score=None
        ))
    return ChatRAGResult(context=chunks)

__all__ = ["retrieve_chat_context"]
