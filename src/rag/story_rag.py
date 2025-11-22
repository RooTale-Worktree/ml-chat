from __future__ import annotations
from typing import List, Optional, Dict
from pathlib import Path
from functools import lru_cache 

from llama_index.core import StorageContext, load_index_from_storage, VectorStoreIndex
from llama_index.core.schema import NodeWithScore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from src.config.config import settings


# 1. '스토리 인덱스' (Story RAG)가 저장된 기본 폴더
STORY_INDEX_BASE_DIR = "data/story_indexes"

# 2. Cached embedding model (lazy loading)
_EMBED_MODEL = None

def _get_embed_model():
    """Lazy load and cache the embedding model."""
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        try:
            _EMBED_MODEL = HuggingFaceEmbedding(model_name=settings.embed_model_name)
            print(f"[Story RAG] Embedding model loaded: {settings.embed_model_name}")
        except Exception as e:
            print(f"경고: HuggingFace 임베딩 모델 로드 실패. {e}")
            raise
    return _EMBED_MODEL

# --- 인덱스 로더 함수 ---

@lru_cache(maxsize=10) # <-- 최근 사용한 10개의 스토리 인덱스를 메모리에 캐싱
def load_story_index(story_title: str) -> Optional[VectorStoreIndex]:
    """
    'story_title'을 기반으로 사전 구축된 '스토리 인덱스'를 로드합니다.
    """
    try:
        # Set embedding model (lazy load on first call)
        Settings.embed_model = _get_embed_model()
        
        index_dir = Path(STORY_INDEX_BASE_DIR) / story_title
        if not index_dir.exists():
            print(f"경고: Story 인덱스를 찾을 수 없습니다. (경로: {index_dir})")
            return None
            
        storage_context = StorageContext.from_defaults(persist_dir=str(index_dir))
        index = load_index_from_storage(storage_context)
        print(f"Story 인덱스 로드 성공: {story_title}")
        return index
    except Exception as e:
        print(f"Story 인덱스 '{story_title}' 로드 중 오류 발생: {e}")
        return None

# --- 메인 검색 함수 ---

def retrieve_story_context(
    story_title: str,        
    user_query: str
) -> List:
    
    all_nodes: List[NodeWithScore] = []
    results: List[Dict] = []

    # 1.검색
    story_index = load_story_index(story_title)
    
    if story_index:
        retriever = story_index.as_retriever()
        retrieved_nodes = retriever.retrieve(user_query)
        all_nodes.extend(retrieved_nodes)
    else:
        print(f"'{story_title}' 인덱스가 로드되지 않아 RAG 검색을 건너뜁니다.")
        return results

    # 2.결과 변환 (Node -> List[Dict])
    for i, node_with_score in enumerate(all_nodes):
        node = node_with_score.node
        results.append({
            "text": node.get_content(),
            "score": node_with_score.score 
        })
    return results

__all__ = ["retrieve_story_context"]