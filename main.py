#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
character_chat.py
- API 내부에서 호출할 "모듈" 관점의 구현
- 공개 모델: PygmalionAI/pygmalion-7b
- 핵심 공개 함수: chat(payload: dict) -> dict
- __main__에서는 루트의 sample_persona.json으로 간단 테스트
"""

from __future__ import annotations
import json
import os
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------
# 기본 설정
# ---------------------------
_DEFAULT_MODEL_ID = "PygmalionAI/pygmalion-6b"
_RECENT_TURNS_TO_KEEP = 12
_STOP_STRINGS = ["\nUser:", "\nUSER:", "\n사용자:", f"\n"]  # 간단 스톱 규칙

# 모델 캐시(프로세스 내 공유)
_MODEL = None
_TOKENIZER = None
_MODEL_ID = None


# ---------------------------
# 유틸
# ---------------------------
def _load_model_and_tokenizer(model_id: str):
    """모델/토크나이저를 전역 캐시로 로드. 이미 로드되어 있으면 재사용."""
    global _MODEL, _TOKENIZER, _MODEL_ID
    if _MODEL is not None and _TOKENIZER is not None and _MODEL_ID == model_id:
        return _MODEL, _TOKENIZER

    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        device_map = "auto"
    else:
        dtype = torch.float32
        device_map = None

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map=device_map
    )
    model.eval()

    _MODEL, _TOKENIZER, _MODEL_ID = model, tokenizer, model_id
    return model, tokenizer


def _build_system_prompt(p: Dict) -> str:
    """Persona 사양으로 System/Examples 블록 구성."""
    constraints = "; ".join(p.get("constraints", []))
    system = (
        f"<<SYSTEM>>\n"
        f"You are roleplaying as the character below. Always stay in character and obey constraints.\n\n"
        f"[Character]\n"
        f"Name: {p['name']}\n"
        f"Persona: {p['persona']}\n"
        f"Scenario: {p['scenario']}\n"
        f"Speaking style: {p['speaking_style']}\n"
        f"Constraints: {constraints}\n"
        f"<<END_SYSTEM>>\n"
    )
    # 예시 대화(톤 고정 보조)
    ex_lines = []
    for turn in p.get("example_dialogue", []):
        role = (turn.get("role") or "").lower()
        txt = turn.get("text") or ""
        if role == "character":
            ex_lines.append(f"{p['name']}: {txt}")
        else:
            ex_lines.append(f"User: {txt}")
    if ex_lines:
        system += f"\n<<EXAMPLES>>\n" + "\n".join(ex_lines) + "\n<<END_EXAMPLES>>\n"
    return system


def _assemble_prompt(system_prompt: str, history: List[Dict], user_text: str, char_name: str) -> str:
    """대화 프롬프트 생성: 최근 N턴만 유지 + 이번 user 발화 + 캐릭터 응답 유도."""
    short_hist = history[-_RECENT_TURNS_TO_KEEP:]
    dlg = []
    for h in short_hist:
        if h["role"] == "user":
            dlg.append(f"User: {h['text']}")
        else:
            dlg.append(f"{char_name}: {h['text']}")
    convo = "\n".join(dlg)
    prompt = (
        f"{system_prompt}\n"
        f"<<CONVERSATION>>\n"
        f"{convo}\n"
        f"User: {user_text}\n"
        f"{char_name}:"
    )
    return prompt


def _postprocess(generated: str) -> Tuple[str, str]:
    """간단 스톱 문자열 기반 자르기 + stop_reason 반환."""
    cut_idx = len(generated)
    reason = "eos"
    for s in _STOP_STRINGS:
        idx = generated.find(s)
        if idx != -1:
            cut_idx = min(cut_idx, idx)
            reason = "stop_string"
    gen = generated[:cut_idx].strip()
    return gen, reason


# ---------------------------
# 공개 함수: chat
# ---------------------------
def chat(payload: Dict) -> Dict:
    """
    persona + (history) + message를 입력받아 캐릭터 답변을 반환.

    Args:
        payload: dict with keys:
            - persona: dict (필수)
            - message: str  (필수)
            - history: List[dict] (선택)
            - gen: dict (선택)
            - model: dict (선택)

    Returns:
        dict: {
           "reply": str,
           "meta": {...}
        }
    """
    # -------- 파라미터 파싱 --------
    persona = payload.get("persona") or {}
    message = payload.get("message") or ""
    history = payload.get("history") or []
    gen = payload.get("gen") or {}
    model_opts = payload.get("model") or {}

    # 필수 키 검증(간단)
    for k in ["name", "persona", "scenario", "speaking_style"]:
        if k not in persona:
            raise ValueError(f"persona missing key: {k}")
    if not isinstance(history, list):
        raise ValueError("history must be a list of {role,text} dicts")
    if not message:
        raise ValueError("message is required")

    model_id = model_opts.get("model_id", _DEFAULT_MODEL_ID)
    max_new_tokens = int(gen.get("max_new_tokens", 256))
    temperature = float(gen.get("temperature", 0.8))
    top_p = float(gen.get("top_p", 0.9))
    repetition_penalty = float(gen.get("repetition_penalty", 1.1))

    # -------- 모델 로드(캐시) --------
    model, tokenizer = _load_model_and_tokenizer(model_id)

    # -------- 프롬프트 구성 --------
    system_prompt = _build_system_prompt(persona)
    prompt = _assemble_prompt(system_prompt, history, message, persona["name"])

    # -------- 생성 --------
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    if torch.cuda.is_available():
        input_ids = input_ids.to(model.device)

    with torch.no_grad():
        out_ids = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            eos_token_id=tokenizer.eos_token_id
        )

    full_text = tokenizer.decode(out_ids[0], skip_special_tokens=True)
    generated = full_text[len(prompt):]
    reply, stop_reason = _postprocess(generated)

    # -------- 결과 패키징 --------
    result = {
        "reply": reply if reply else "(…생각 중…)",
        "meta": {
            "model_id": model_id,
            "prompt_tokens": None,          # 필요시 tokenizer로 추정치 계산 가능
            "completion_tokens": None,
            "stop_reason": stop_reason,
            "used_history_turns": min(len(history), _RECENT_TURNS_TO_KEEP),
        }
    }
    return result


# ---------------------------
# __main__: 간단 테스트
# 루트의 sample_persona.json을 읽고 한 턴 실행
# ---------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Persona chat (module test)")
    parser.add_argument("--persona", type=str, default="sample_persona.json",
                        help="루트 디렉토리의 persona JSON 경로")
    parser.add_argument("--message", type=str, default="오늘 공부 계획 어떻게 잡을까?",
                        help="사용자 입력 문장")
    parser.add_argument("--model-id", type=str, default=_DEFAULT_MODEL_ID,
                        help="Hugging Face model id")
    args = parser.parse_args()

    if not os.path.exists(args.persona):
        raise FileNotFoundError(f"sample persona not found: {args.persona}")

    with open(args.persona, "r", encoding="utf-8") as f:
        persona_obj = json.load(f)

    payload = {
        "persona": persona_obj,
        "message": args.message,
        "history": [],       # 첫 턴이면 빈 리스트
        "gen": {
            "max_new_tokens": 256,
            "temperature": 0.8,
            "top_p": 0.9
        },
        "model": {
            "model_id": args.model_id
        }
    }
    out = chat(payload)
    print(f"\n[{persona_obj['name']}] {out['reply']}\n")