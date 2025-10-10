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

    sample = {
        "persona": persona,
        "history": history,
        "story": story,
        **others
    }
    out = handler({"input": sample})
    from pprint import pprint
    pprint(out)
