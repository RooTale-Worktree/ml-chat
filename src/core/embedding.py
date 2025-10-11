"""
Sentence embedding utilities for RAG.

Model: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 (384d)
- Suited for multilingual (incl. Korean) sentence embeddings
- Returns L2-normalized vectors by default (cosine ~ dot product)
"""
from __future__ import annotations
from typing import Iterable, List
from functools import lru_cache

try:
    from sentence_transformers import SentenceTransformer
except Exception as e:  # pragma: no cover
    SentenceTransformer = None  # type: ignore

_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

@lru_cache(maxsize=1)
def get_embedder():
    if SentenceTransformer is None:
        raise ImportError("sentence-transformers is not installed. Add it to requirements.txt")
    return SentenceTransformer(_MODEL_NAME)


def embed_text(text: str, normalize: bool = True) -> List[float]:
    """Embed a single text into a 384-dim vector.

    Args:
        text: Input sentence.
        normalize: Whether to L2-normalize the output embedding.

    Returns:
        List[float]: Embedding vector (length 384).
    """
    model = get_embedder()
    vec = model.encode([text], normalize_embeddings=normalize)[0]
    return vec.tolist()


def embed_texts(texts: Iterable[str], normalize: bool = True) -> List[List[float]]:
    """Batch embed multiple texts.

    Args:
        texts: Iterable of sentences.
        normalize: Whether to L2-normalize the output embeddings.

    Returns:
        List of embedding vectors.
    """
    model = get_embedder()
    vectors = model.encode(list(texts), normalize_embeddings=normalize)
    return [v.tolist() for v in vectors]

__all__ = ["get_embedder", "embed_text", "embed_texts"]
