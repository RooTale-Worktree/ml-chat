"""
Story retrieval stub.
Loads mock story JSON and slices first K paragraphs.
Replace later with vector similarity search.
"""
from __future__ import annotations
from pathlib import Path
import json
from typing import List

from src.schemas.request import StoryEvent
from src.schemas.rag import RAGChunk, StoryRAGResult
from src.config.config import settings

from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def retrieve_story_context(story: List[StoryEvent], user_query: str) -> StoryRAGResult:
    
    docs: List[Document] = []
    for ev in story:
        docs.append(Document(
            page_content=ev.content,
            metadata={
                "story_id": str(ev.story_id),
                "chunk_no": ev.chunk_no,
                "chunk_type": ev.chunk_type,
                "timestamp": ev.timestamp.isoformat() if ev.timestamp else None,
                **ev.meta
            }
        ))
    
    embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sbert-nli")
    vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings)
    retriever = vectorstore.as_retriever()
    retrieved_documents = retriever.invoke(user_query)

    chunks: List[RAGChunk] = []
    for i, doc in enumerate(retrieved_documents):
        chunks.append(RAGChunk(
            id=f"story-{i}", 
            source=doc.metadata.get("source", "story_document"),
            text=doc.page_content, 
            score=None 
        ))
    return StoryRAGResult(context=chunks)

__all__ = ["retrieve_story_context"]
