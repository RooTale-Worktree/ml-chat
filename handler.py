from __future__ import annotations
"""RunPod serverless handler entrypoint.

RunPod expects a handler(event) -> dict style function.
Each event should include a JSON payload conforming to ChatRequest schema.
"""
from src.service.orchestrator import handle_chat


def handler(event):  # RunPod style
    # event may have {'input': {...}} depending on client
    payload = event.get('input') if isinstance(event, dict) else event
    if not isinstance(payload, dict):
        raise ValueError("Invalid event payload; expected dict or {'input': dict}")
    return handle_chat(payload)

# For local manual test
if __name__ == "__main__":
    from src.core.persona import load_persona
    persona = load_persona()
    sample = {
        "message": "안녕? 오늘 기분 어때?",
        "history": [],
        "persona": persona.model_dump(),
        "gen": {},
        "model": {}
    }
    out = handler({"input": sample})
    from pprint import pprint
    pprint(out)
