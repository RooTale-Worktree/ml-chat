#!/usr/bin/env python3
"""Stub script to build vector indexes (chat/story) for future RAG.
Currently just creates a placeholder file in data/indexes.
"""
from pathlib import Path

INDEX_DIR = Path("data/indexes")


def main():
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    placeholder = INDEX_DIR / "README.txt"
    placeholder.write_text("Placeholder for vector indexes. Replace with FAISS/Chroma build.", encoding="utf-8")
    print(f"Wrote placeholder index file at {placeholder}")

if __name__ == "__main__":
    main()
