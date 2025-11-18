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
import os
from pathlib import Path

from src.orchestrator import handle_chat
from src.llm.get_llm import get_llm
from src.schemas.request import ModelConfig
import numpy as np


# ===== RunPod Serverless optimization: model pre-loading =====
# Load model once at container start (optimize cold start)
_PRELOADED_LLM = None

def _init_model():
    """Load model once at container start (optimize cold start)"""
    global _PRELOADED_LLM
    if _PRELOADED_LLM is None:
        print("[INIT] Loading model for RunPod serverless...")
        model_name = os.getenv("MLCHAT_MODEL_NAME", "gpt-oss-20b")
        
        # vLLM 설정
        model_cfg = ModelConfig(
            name=model_name,
            tensor_parallel_size=int(os.getenv("TENSOR_PARALLEL_SIZE", "1")),
            gpu_memory_utilization=float(os.getenv("GPU_MEMORY_UTILIZATION", "0.9")),
        )
        
        _PRELOADED_LLM = get_llm(model_name, model_cfg)
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
        event: {"input": {...}} 형식의 요청
        
    Returns:
        ChatResponse dict
    """
    payload = event.get('input') if isinstance(event, dict) else event
    if not isinstance(payload, dict):
        raise ValueError("Invalid event payload; expected dict or {'input': dict}")
    
    # Use pre-loaded model
    llm = _init_model()
    return handle_chat(payload, llm_instance=llm)


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
    parser.add_argument("--message", type=str, default="안녕, 오늘 기분 어때?", help="user message")
    parser.add_argument("--others", type=str,
                        default="./data/mock/sample_request_gpt_oss.json", help="other request metas")
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
        "message": args.message,
        **others
    }

    # call handler function
    out = handler({"input": sample})

    # 출력: 모델 응답 텍스트와 타이밍 정보
    try:
        # response_contents는 List이므로 첫 번째 아이템의 content를 출력
        first_content = (
            out.get("response_contents", [{}])[0].get("content")
            if isinstance(out, dict) else None
        )
        print("\n[Model Reply]\n" + (first_content or "<no content>"))
        # timing 전체 출력 (ms 단위 세부 항목 포함)
        timing = out.get("timing", {}) if isinstance(out, dict) else {}
        print("\n[Timing ms]\n" + json.dumps(timing, ensure_ascii=False, indent=2))
    except Exception as e:
        print(f"[Print Error] {e}")
    
    from pprint import pprint
    print("\n[Full Response]")
    pprint(out)
