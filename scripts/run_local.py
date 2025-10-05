#!/usr/bin/env python3
"""Local end-to-end test using orchestrator + mock llm."""
from pathlib import Path
import json
from src.core.persona import load_persona
from src.service.orchestrator import handle_chat

def main():
    persona = load_persona()
    req = {
        "message": "캐릭터, 오늘 모험 계획 어때?",
        "history": [
            {"role": "user", "text": "안녕"},
            {"role": "character", "text": "반가워."}
        ],
        "persona": persona.model_dump(),
        "gen": {},
        "model": {}
    }
    result = handle_chat(req)
    print("Reply:", result["reply"])  # type: ignore

if __name__ == "__main__":
    main()
