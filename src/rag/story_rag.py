from __future__ import annotations
"""Story retrieval stub.
Loads mock story JSON and slices first K paragraphs.
Replace later with vector similarity search.
"""
from pathlib import Path
import json
from typing import List
from src.core.schemas import RAGChunk, StoryRAGResult
from src.config.config import settings

_STORY_PATHS = [Path("data/mock/story.json"), Path("mock_data/mock_story.json"), Path("story.json")]


def _load_story() -> List[str]:
    for p in _STORY_PATHS:
        if p.exists():
            with p.open("r", encoding="utf-8") as f:
                data = json.load(f)
            # Expecting {"paragraphs": ["..."]}
            paras = data.get("paragraphs") or data.get("story") or []
            if isinstance(paras, list):
                return [str(x) for x in paras]
    return []


def retrieve_story_context() -> StoryRAGResult:
    paragraphs = _load_story()[: settings.story_k]
    chunks: List[RAGChunk] = []
    for i, para in enumerate(paragraphs):
        chunks.append(RAGChunk(
            id=f"story-{i}",
            source="story",
            text=para,
            score=None
        ))
    return StoryRAGResult(context=chunks)

__all__ = ["retrieve_story_context"]
