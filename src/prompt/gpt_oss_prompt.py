"""
gpt-oss-20b (Base Model)에 최적화된 프롬프트 빌더.
복잡한 지시어 태그 대신, 문맥(Context)과 예시(Few-shot)를 통해
역할극 패턴을 학습시키는 간단한 텍스트 형식으로 프롬프트를 생성합니다.
"""
from __future__ import annotations
from typing import List

from src.schemas.request import DialogueTurn
from src.schemas.rag import PromptBuildInput, PromptBuildOutput, RAGChunk

# --- (삭제) ---
# SYSTEM_HEADER 및 복잡한 명령어는 베이스 모델이 이해하지 못하므로 제거합니다.
# -----------------


def _format_context(label: str, chunks: List[RAGChunk]) -> str:
    """
    RAG 청크를 단순한 텍스트 참고 자료로 포맷합니다.
    """
    if not chunks:
        return ""  # 컨텍스트가 없으면 아무것도 반환하지 않습니다.
    
    # 모델이 이해할 수 있도록 간단한 헤더와 목록으로 변환
    lines = [f"\n# {label} (참고 정보)"]
    lines.extend(f"- {chunk.text}" for chunk in chunks if chunk.text)
    return "\n".join(lines) + "\n"


def _format_examples(char_name: str, turns: List[DialogueTurn]) -> str:
    """
    대화 예시를 'User'와 '캐릭터'간의 대본 형식으로 포맷합니다.
    이것이 베이스 모델에게 말투를 학습시키는 가장 효과적인 방법입니다.
    """
    if not turns:
        return ""
        
    lines = ["\n# 대화 예시"]
    for t in turns:
        # 'character' 역할을 실제 캐릭터 이름으로 변경
        role = char_name if t.role == "character" else "User"
        lines.append(f"{role}: {t.content}")
    return "\n".join(lines) + "\n"


# --- (삭제) ---
# _history_to_harmony 함수는 불필요한 태그를 사용하므로 제거합니다.
# -----------------


def build_prompt(data: PromptBuildInput) -> PromptBuildOutput:
    """
    gpt-oss-20b 베이스 모델을 위한 단순한 텍스트 프롬프트를 생성합니다.
    """
    persona = data.persona
    char_name = persona.character_name

    # 프롬프트 라인을 담을 리스트
    prompt_lines: List[str] = []

    # 1. (가장 중요) 시스템/개발자 명령어 대신, 간단한 '역할 정의'를 제공합니다.
    #    "~인 척 해" (Roleplay) 보다는 "~이다" (Context) 방식이 효과적입니다.
    prompt_lines.append(f"--- 대화 시뮬레이션 ---")
    prompt_lines.append(f"당신은 '{char_name}'입니다.")
    prompt_lines.append(f"역할: {persona.persona}")
    prompt_lines.append(f"상황: {persona.scenario}")
    prompt_lines.append(f"말투: {persona.speaking_style}")
    constraints = "; ".join(persona.constraints) if persona.constraints else "없음"
    prompt_lines.append(f"제약: {constraints}")
    prompt_lines.append(f"---")

    # 2. RAG 컨텍스트 (스토리, 대화 기억)를 간단한 참고 자료로 추가합니다.
    prompt_lines.append(_format_context("스토리", data.story_context))
    prompt_lines.append(_format_context("대화 기억", data.chat_context))

    # 3. 대화 예시 (Few-shot)를 추가합니다.
    #    베이스 모델은 이 예시를 보고 말투와 패턴을 학습합니다.
    prompt_lines.append(_format_examples(char_name, persona.example_dialogue))

    # 4. 최근 대화 기록을 태그 없이, 예시와 동일한 대본 형식으로 추가합니다.
    if data.recent_chat:
        prompt_lines.append("\n# 최근 대화")
        for turn in data.recent_chat:
            # 'assistant' 역할을 실제 캐릭터 이름으로 변경
            role = char_name if turn.role == "assistant" else "User"
            prompt_lines.append(f"{role}: {turn.content}")
        prompt_lines.append("") # 대화 구분을 위해 한 줄 띄움

    # 5. 사용자의 마지막 메시지를 추가합니다.
    prompt_lines.append(f"User: {data.user_message}")
    
    # 6. 모델이 이어서 생성할 부분(캐릭터의 응답)을 표시합니다.
    prompt_lines.append(f"{char_name}:")

    # 모든 라인을 합쳐 최종 프롬프트를 생성합니다.
    # 빈 라인(None 이나 "")은 제거합니다.
    prompt = "\n".join(line for line in prompt_lines if line is not None)

    meta = {
        "chat_ctx_count": len(data.chat_context),
        "story_ctx_count": len(data.story_context),
    }
    return PromptBuildOutput(prompt=prompt, meta=meta)


__all__ = ["build_prompt"]