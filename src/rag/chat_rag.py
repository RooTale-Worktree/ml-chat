"""
Chat (recent dialogue) retrieval stub.
For now, returns last N user/character turns as pseudo-context.
Later can be replaced by semantic search over dialogue memory store.
"""
from __future__ import annotations
from typing import List, Union

from src.schemas.request import Message
from src.schemas.rag import DialogueTurn, RAGChunk, ChatRAGResult
from src.config.config import settings

_DEF_SOURCE = "chat_history"

def retrieve_chat_context(history: List[Union[DialogueTurn, Message]]) -> ChatRAGResult:
    # Normalize to DialogueTurn
    norm: List[DialogueTurn] = []
    for h in history:
        if isinstance(h, Message):
            norm.append(h.to_dialogue_turn())
        else:
            norm.append(h)
    turns = norm[-settings.chat_k:]
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
