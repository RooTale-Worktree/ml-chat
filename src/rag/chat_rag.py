"""
Chat (recent dialogue) retrieval stub.
"""
from __future__ import annotations
from typing import List
from datetime import datetime, timezone, timedelta
import math

from src.schemas.request import Message, ChatRAGConfig
from src.schemas.rag import DialogueTurn, RAGChunk, ChatRAGResult
from src.config.config import settings

_DEF_SOURCE = "chat_history"

def _recent_fallback(history: List[Message], top_k: int) -> ChatRAGResult:
    turns = [h.to_dialogue_turn() for h in history[-top_k:]]
    chunks = [
        RAGChunk(
            id=f"chat-{m.chat_id}-{m.seq_no}",
            source=_DEF_SOURCE,
            text=f"{t.role}: {t.content}",
            score=None,
        )
        for t, m in zip(turns, history[-top_k:])
    ]
    return ChatRAGResult(context=chunks)


def retrieve_chat_context(history: List[Message],
                          chat_rag_config: ChatRAGConfig,
                          query_embedding: List[float] | None = None) -> ChatRAGResult:
    """
    args:
        history: List[Message] - chat history messages from oldest to newest
        chat_rag_config: ChatRAGConfig - configuration for chat RAG retrieval
        query_embedding: List[float] | None - embedding vector of the current user message

    returns:
        ChatRAGResult - retrieved chat context as RAGChunks

    If query_embedding is None or no messages have embeddings, fall back to recent K messages.
    Uses PyTorch if available, otherwise NumPy for vector similarity search.
    """

    items: List[Message] = history
    if chat_rag_config.history_time_window_min:
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=chat_rag_config.history_time_window_min)
        items = [m for m in items if m.timestamp and m.timestamp >= cutoff]

    cand_msgs: List[Message] = []
    cand_vecs: List[List[float]] = []
    for m in items:
        if m.embedding is not None:
            cand_msgs.append(m)
            cand_vecs.append(m.embedding)

    if (not cand_msgs) or (not query_embedding):
        return _recent_fallback(items, chat_rag_config.history_top_k)
    
    try:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"

        H = torch.tensor(cand_vecs, dtype=torch.float32, device=device)         # (N, d)
        q = torch.tensor(query_embedding, dtype=torch.float32, device=device)   # (d,)

        sims = torch.mv(H, q)   # (N,)
        if chat_rag_config.threshold is not None:
            mask = sims < chat_rag_config.threshold
            sims = sims.masked_fill(mask, -math.inf)

        top_k = max(1, min(chat_rag_config.top_k_history, sims.numel()))
        vals, idx = torch.topk(sims, top_k)

        context: List[RAGChunk] = []
        for r, v in zip(idx.tolist(), vals.tolist()):
            if not math.isfinite(v):
                continue
            m = cand_msgs[r]
            label = m.character_name or m.role
            context.append(
                RAGChunk(
                    id=f"chat-{m.chat_id}-{m.seq_no}",
                    source=_DEF_SOURCE,
                    text=f"{label}: {m.content}",
                    score=float(v),
                )
            )
        
        if not context:
            return _recent_fallback(items, chat_rag_config.history_top_k)
        else:
            return ChatRAGResult(context=context)
        
    except Exception:
        import numpy as np

        H = np.asarray(cand_vecs, dtype=np.float32)         # (N, d)
        q = np.asarray(query_embedding, dtype=np.float32)   # (d,)

        sims = H @ q   # (N,)
        if chat_rag_config.threshold is not None:
            sims = np.where(sims >= chat_rag_config.threshold, sims, -np.inf)

        top_k = max(1, min(chat_rag_config.top_k_history, sims.shape[0]))
        part_idx = np.argpartition(sims, -top_k)[-top_k:]
        sel = part_idx[np.argsort(-sims[part_idx])]

        context: List[RAGChunk] = []
        for r in sel:
            v = sims[r]
            if not np.isfinite(v):
                continue
            m = cand_msgs[r]
            label = m.character_name or m.role
            context.append(
                RAGChunk(
                    id=f"chat-{m.chat_id}-{m.seq_no}",
                    source=_DEF_SOURCE,
                    text=f"{label}: {m.content}",
                    score=float(v),
                )
            )

        if not context:
            return _recent_fallback(items, chat_rag_config.history_top_k)
        else:
            return ChatRAGResult(context=context)
        

__all__ = ["retrieve_chat_context"]
