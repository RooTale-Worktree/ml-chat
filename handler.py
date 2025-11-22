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
import argparse, json, os
from pathlib import Path

from src.config.config import settings
from src.orchestrator import handle_chat
from src.llm.get_llm import get_llm
from src.schemas.request import ModelConfig
import numpy as np


# ===== RunPod Serverless optimization: model pre-loading =====
# Load model once at container start (optimize cold start)
_PRELOADED_LLM = None
_PRELOADED_MODEL_NAME = None

def _init_model(model_cfg: dict):
    """Load model once at container start (optimize cold start)"""
    
    global _PRELOADED_LLM, _PRELOADED_MODEL_NAME
    
    # Determine requested model name
    model_name = model_cfg.get("model_name", None)
    if not model_name:
        model_name = settings.default_model_name
    
    # If already loaded and same model, return cached instance
    if _PRELOADED_LLM is not None and _PRELOADED_MODEL_NAME == model_name:
        print(f"[INIT] Using cached model: {model_name}")
        return _PRELOADED_LLM
    
    # Load new model
    print(f"[INIT] Loading model for RunPod serverless: {model_name}")
    
    _PRELOADED_LLM = get_llm(model_name, model_cfg)
    _PRELOADED_MODEL_NAME = model_name
    print(f"[INIT] Model loaded: {model_name}")
    
    return _PRELOADED_LLM

# If in RunPod environment, initialize at module load time
if os.getenv("RUNPOD_ENDPOINT_ID"):
    _init_model()


# Entrypoint
def handler(event):
    """
    RunPod serverless handler.
    Args:
        event: {"input": {...}}
    Returns:
        ChatResponse dict or error dict
    """
    try:
        print("[HANDLER] Received event")
        payload = event.get('input') if isinstance(event, dict) else event
        if not isinstance(payload, dict):
            raise ValueError("Invalid event payload; expected dict or {'input': dict}")
        
        # Use pre-loaded model
        print("[HANDLER] Initializing model...")
        model_cfg = payload.get("model_cfg", {})

        print(model_cfg)
        llm = _init_model(model_cfg)
        return handle_chat(payload, llm_instance=llm)
    
    except ValueError as e:
        # Input validation errors
        error_msg = f"Validation error: {str(e)}"
        print(f"[HANDLER ERROR] {error_msg}")
        return {
            "error": error_msg,
            "error_type": "ValidationError",
            "status": "failed"
        }
    
    except Exception as e:
        # Catch all other errors (orchestrator, model loading, etc.)
        error_msg = f"Internal error: {str(e)}"
        print(f"[HANDLER ERROR] {error_msg}")
        import traceback
        traceback.print_exc()
        return {
            "error": error_msg,
            "error_type": type(e).__name__,
            "status": "failed"
        }


# ===== For local manual test =====
def _load_json(path: str):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Json file not found: {path}")
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--persona", type=str,
                        default="./data/mock/sample_persona.json", help="mock up persona")
    parser.add_argument("--chat_history", type=str,
                        default="./data/mock/sample_chat_history.json", help="mock up message chat history")
    parser.add_argument("--message", type=str, default="안녕, 오늘 기분 어때?", help="user message")
    parser.add_argument("--others", type=str,
                        default="./data/mock/sample_request_gpt_oss.json", help="other request metas")
    args = parser.parse_args()

    persona = _load_json(args.persona)
    chat_history = _load_json(args.chat_history)
    others = _load_json(args.others)

    # structure sample request
    sample = {
        "persona": persona,
        "chat_history": chat_history,
        "message": args.message,
        **others
    }

    # call handler function
    out = handler({"input": sample})

    # print response content & timing
    from pprint import pprint
    print("\n[Full Response]")
    pprint(out)