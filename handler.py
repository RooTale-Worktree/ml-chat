"""
RunPod serverless의 handler entrypoint.
각각의 event는 ChatRequest schema를 따르는 JSON payload를 포함해야 합니다.

local manual test 사용법:
    python -m handler
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from src.orchestrator import handle_chat
import numpy as np


# Entrypoint
def handler(event):
    payload = event.get('input') if isinstance(event, dict) else event
    if not isinstance(payload, dict):
        raise ValueError("Invalid event payload; expected dict or {'input': dict}")
    return handle_chat(payload)


# For local manual test
def load_json(path: str):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Json file not found: {path}")
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--persona", type=str,
                        default="./data/mock/sample_persona.json", help="mock up persona")
    parser.add_argument("--history", type=str,
                        default="./data/mock/sample_history.json", help="mock up message history")
    parser.add_argument("--story", type=str,
                        default="./data/mock/sample_story.json", help="mock up story")
    parser.add_argument("--others", type=str,
                        default="./data/mock/sample_request.json", help="other request metas")
    args = parser.parse_args()

    persona = load_json(args.persona)
    history = load_json(args.history)
    story = load_json(args.story)
    others = load_json(args.others)

    # Materialize embeddings from embedding_ref (if present)
    try:
        if isinstance(history, list):
            for msg in history:
                if not isinstance(msg, dict):
                    continue
                ref = msg.get("embedding_ref")
                if isinstance(ref, str) and ref:
                    try:
                        vec = np.load(ref).astype(float).tolist()
                        msg["embedding"] = vec
                    except Exception:
                        # Leave as-is if loading fails
                        msg["embedding"] = None
                        pass
                    msg.pop("embedding_ref", None)
    except Exception:
        # Non-fatal; proceed without embedding materialization
        pass

    sample = {
        "persona": persona,
        "history": history,
        "story": story,
        **others
    }
    out = handler({"input": sample})
    from pprint import pprint
    pprint(out)
