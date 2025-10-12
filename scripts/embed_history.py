"""
Fill missing embeddings for mock chat history.

Reads a JSON file containing a List[Message] (see src/schemas/request.Message).
For each message where embedding metadata is missing (embedding/model/dim/etag is None),
compute the embedding with src.core.embedding.embed_text and write back to the same file.

Usage:
  python scripts/embed_history.py                 # uses default path data/mock/sample_history.json
  python scripts/embed_history.py --path PATH     # specify a different JSON path
  python scripts/embed_history.py --dry-run       # show what would change without writing
  python scripts/embed_history.py --recompute     # recompute even if embeddings exist
"""
from __future__ import annotations
from pathlib import Path
import argparse
import json
import hashlib
from typing import Any, Dict, List, Optional

from src.core.embedding import embed_text
import numpy as np


DEFAULT_PATH = Path("data/mock/sample_history.json")
VECTOR_DIR = Path("data/mock/vector")
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


# def _needs_embedding(msg: Dict[str, Any], force: bool) -> bool:
#     if force:
#         return True
#     return (
#         msg.get("embedding") is None
#         or msg.get("embedding_dim") is None
#         or msg.get("embedding_model") is None
#         or msg.get("embedding_etag") is None
#     )


def _etag_for(text: str, model: str) -> str:
    h = hashlib.sha1()
    h.update(model.encode("utf-8"))
    h.update(b"\x00")
    h.update(text.strip().encode("utf-8"))
    return f"sha1:{h.hexdigest()}"


def process(path: Path, dry_run: bool = False, force: bool = False) -> Dict[str, int]:
    if not path.exists():
        raise FileNotFoundError(f"History JSON not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Expected a JSON array of Message objects")

    updated = 0
    skipped = 0

    # Ensure vector directory exists when writing
    if not dry_run:
        VECTOR_DIR.mkdir(parents=True, exist_ok=True)

    for msg in data:
        if not isinstance(msg, dict):
            skipped += 1
            continue
        content = str(msg.get("content") or "").strip()
        if not content:
            skipped += 1
            continue

        etag: str = msg.get("embedding_etag") or _etag_for(content, EMBED_MODEL)
        npy_path = VECTOR_DIR / f"{etag}.npy"

        ref_path: Optional[str] = msg.get("embedding_ref") if isinstance(msg.get("embedding_ref"), str) else None
        has_ref_file = Path(ref_path).exists() if ref_path else False

        # Case 1: already has a valid ref (and not forcing recompute)
        if ref_path and has_ref_file and not force:
            # Ensure metadata fields exist; set or correct them if missing
            meta_changed = False
            if not msg.get("embedding_model"):
                msg["embedding_model"] = EMBED_MODEL
                meta_changed = True
            if not msg.get("embedding_etag"):
                msg["embedding_etag"] = etag
                meta_changed = True
            if not msg.get("embedding_dim"):
                try:
                    if not dry_run:
                        vec = np.load(ref_path)
                        msg["embedding_dim"] = int(vec.shape[-1]) if hasattr(vec, "shape") else len(vec.tolist())
                    else:
                        msg["embedding_dim"] = 384  # best-effort default in dry-run
                    meta_changed = True
                except Exception:
                    # Could not load; leave dim as-is
                    pass
            if meta_changed:
                updated += 1
            else:
                skipped += 1
            continue

        # Case 2: compute (no ref or forcing recompute)
        vec_to_store: List[float] = embed_text(content)
        if not dry_run:
            np.save(npy_path, np.array(vec_to_store, dtype=np.float32))
        msg["embedding_ref"] = str(npy_path)
        msg["embedding_dim"] = len(vec_to_store)
        msg["embedding_model"] = EMBED_MODEL
        msg["embedding_etag"] = etag
        updated += 1

    if not dry_run:
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        tmp.replace(path)

    return {"updated": updated, "skipped": skipped, "total": len(data)}


def main():
    ap = argparse.ArgumentParser(description="Embed mock chat history in-place")
    ap.add_argument("--path", type=str, default=str(DEFAULT_PATH), help="Path to history JSON list")
    ap.add_argument("--dry-run", action="store_true", help="Do not write file; just report")
    ap.add_argument("--recompute", action="store_true", help="Recompute even if fields already exist")
    args = ap.parse_args()

    stats = process(Path(args.path), dry_run=args.dry_run, force=args.recompute)
    print(f"updated={stats['updated']} skipped={stats['skipped']} total={stats['total']}")


if __name__ == "__main__":
    main()
