"""
Chat (recent dialogue) retrieval stub.
"""
from __future__ import annotations
from typing import List, Dict
from datetime import datetime, timezone, timedelta
import math

_DEF_SOURCE = "chat_history"

def _recent_fallback(chat_history: List, top_k: int) -> List[Dict]:
    turns = [h.to_dialogue_turn() for h in chat_history[-top_k:]]
    chunks = [
        {
            "id": f"chat-{i}",
            "source": _DEF_SOURCE,
            "text": f"{t.role}: {t.content}",
            "score": None,
        }
        for i, (t, m) in enumerate(zip(turns, chat_history[-top_k:]))
    ]
    return chunks

def retrieve_chat_context(chat_history: List,
                          chat_rag_config: Dict,
                          query_embedding: List[float] | None = None) -> List[Dict]:
    """
    args:
        chat_history: List - chat history messages from oldest to newest
        chat_rag_config: Dict - configuration for chat RAG retrieval
        query_embedding: List[float] | None - embedding vector of the current user message

    returns:
        List[Dict] - retrieved chat context as dictionaries with keys: id, source, text, score

    If query_embedding is None or no messages have embeddings, fall back to recent K messages.
    Uses PyTorch if available, otherwise NumPy for vector similarity search.
    """

    items: List = chat_history
    if chat_rag_config.get("history_time_window_min"):
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=chat_rag_config["history_time_window_min"])
        items = [m for m in items if hasattr(m, "timestamp") and m.timestamp and m.timestamp >= cutoff]

    cand_msgs: List = []
    cand_vecs: List[List[float]] = []
    for m in items:
        if hasattr(m, "embedding") and m.embedding is not None:
            cand_msgs.append(m)
            cand_vecs.append(m.embedding)

    top_k_history = chat_rag_config.get("top_k_history", 6)
    
    if (not cand_msgs) or (not query_embedding):
        return _recent_fallback(items, top_k_history)
    
    try:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"

        H = torch.tensor(cand_vecs, dtype=torch.float32, device=device)         # (N, d)
        q = torch.tensor(query_embedding, dtype=torch.float32, device=device)   # (d,)

        sims = torch.mv(H, q)   # (N,)
        threshold = chat_rag_config.get("threshold")
        if threshold is not None:
            mask = sims < threshold
            sims = sims.masked_fill(mask, -math.inf)

        top_k = max(1, min(top_k_history, sims.numel()))
        vals, idx = torch.topk(sims, top_k)

        context: List[Dict] = []
        for r, v in zip(idx.tolist(), vals.tolist()):
            if not math.isfinite(v):
                continue
            m = cand_msgs[r]
            label = getattr(m, "character_name", None) or getattr(m, "role", "unknown")
            context.append({
                "id": f"chat-{r}",
                "source": _DEF_SOURCE,
                "text": f"{label}: {m.content}",
                "score": float(v),
            })
        
        if not context:
            return _recent_fallback(items, top_k_history)
        else:
            return context
        
    except Exception:
        import numpy as np

        H = np.asarray(cand_vecs, dtype=np.float32)         # (N, d)
        q = np.asarray(query_embedding, dtype=np.float32)   # (d,)

        sims = H @ q   # (N,)
        threshold = chat_rag_config.get("threshold")
        if threshold is not None:
            sims = np.where(sims >= threshold, sims, -np.inf)

        top_k = max(1, min(top_k_history, sims.shape[0]))
        part_idx = np.argpartition(sims, -top_k)[-top_k:]
        sel = part_idx[np.argsort(-sims[part_idx])]

        context: List[Dict] = []
        for r in sel:
            v = sims[r]
            if not np.isfinite(v):
                continue
            m = cand_msgs[r]
            label = getattr(m, "character_name", None) or getattr(m, "role", "unknown")
            context.append({
                "id": f"chat-{r}",
                "source": _DEF_SOURCE,
                "text": f"{label}: {m.content}",
                "score": float(v),
            })

        if not context:
            return _recent_fallback(items, top_k_history)
        else:
            return context
        

__all__ = ["retrieve_chat_context"]
