from __future__ import annotations
import argparse
"""RunPod serverless handler entrypoint.

RunPod expects a handler(event) -> dict style function.
Each event should include a JSON payload conforming to ChatRequest schema.
"""
from src.orchestrator import handle_chat


def handler(event):  # RunPod style
    # event may have {'input': {...}} depending on client
    payload = event.get('input') if isinstance(event, dict) else event
    if not isinstance(payload, dict):
        raise ValueError("Invalid event payload; expected dict or {'input': dict}")
    return handle_chat(payload)


# For local manual test
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str,
                        default="mock_llm", help="model name")
    args = parser.parse_args()

    from src.core.persona import load_persona
    persona = load_persona()
    sample = {
        "message": "안녕? 오늘 기분 어때?",
        "history": [],
        "persona": persona.model_dump(),
        "gen": {},
        "model": {"name": args.model}
    }
    out = handler({"input": sample})
    from pprint import pprint
    pprint(out)
