"""
RunPod serverless의 handler entrypoint.
각각의 event는 ChatRequest schema를 따르는 JSON payload를 포함해야 합니다.

local manual test 사용법:
    python -m handler
    CLI 옵션:
        --persona: mock-up persona 정보가 저장된 파일 경로를 지정합니다.
        --chat_history: mock-up chat history 정보가 저장된 파일 경로를 지정합니다. List[Dict]
        --story: mock-up story 정보가 저장된 파일 경로를 지정합니다. List[Dict]
        --others: 그 외의 ChatRequest 정보가 저장된 mock-up 파일 경로를 지정합니다.
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


"""
For local manual test
"""
def load_json(path: str):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Json file not found: {path}")
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)
    
def load_vector(contents: list):
    # Materialize embeddings from embedding_ref (if present)
    try:
        if isinstance(contents, list):
            for content in contents:
                if (not isinstance(content, dict)) or ("content" not in content):
                   continue
                ref = content.get("embedding_ref")
                if isinstance(ref, str) and ref:
                    try:
                        vec = np.load(ref).astype(float).tolist()
                        content["embedding"] = vec
                    except Exception:
                        # Leave as-is if loading fails
                        content["embedding"] = None
                        pass
                    content.pop("embedding_ref", None)
    except Exception:
        # Non-fatal; proceed without embedding materialization
        pass
    return contents

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--persona", type=str,
                        default="./data/mock/sample_persona.json", help="mock up persona")
    parser.add_argument("--chat_history", type=str,
                        default="./data/mock/sample_chat_history.json", help="mock up message chat history")
    parser.add_argument("--story", type=str,
                        default="./data/mock/sample_story.json", help="mock up story")
    parser.add_argument("--others", type=str,
                        default="./data/mock/sample_request.json", help="other request metas")
    args = parser.parse_args()

    path_chat_history = None if args.chat_history == "" else args.chat_history
    path_story = None if args.story == "" else args.story

    persona = load_json(args.persona)
    chat_history = load_vector(load_json(path_chat_history)) if path_chat_history else []
    story = load_vector(load_json(path_story)) if path_story else []
    others = load_json(args.others)

    # structure sample request
    sample = {
        "persona": persona,
        "chat_history": chat_history,
        "story": story,
        **others
    }

    # call handler function
    out = handler({"input": sample})

    from pprint import pprint
    pprint(out)
