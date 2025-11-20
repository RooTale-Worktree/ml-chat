"""
Prompt builder for SOLAR (Llama-2 Template).
This builder MANUALLY constructs the Llama-2 chat template string
because the tokenizer's apply_chat_template is unreliable for this model.
"""
from __future__ import annotations
from typing import List, Dict

from src.schemas.request import DialogueTurn
from src.schemas.rag import PromptBuildInput, PromptBuildOutput, RAGChunk


def build_prompt(prompt_input: PromptBuildInput) -> PromptBuildOutput:
    """
    Builds a SOLAR-compatible prompt string by manually creating
    the Llama-2 chat template.
    """
    
    persona = prompt_input.persona
    
    rag_lines = []
    if prompt_input.chat_context:
        rag_lines.append("[대화 기억]")
        for chunk in prompt_input.chat_context:
            rag_lines.append(f"- {chunk.text}")
    if prompt_input.story_context:
        rag_lines.append("[스토리 정보]")
        for chunk in prompt_input.story_context:
            rag_lines.append(f"- {chunk.text}")
    rag_context = "\n".join(rag_lines) if rag_lines else "(참고 맥락 없음)"

    constraints = "; ".join(persona.constraints) if persona.constraints else "없음"
    
    system_content = f"""[임무]
당신은 '{persona.character_name}'입니다. 사용자에게 '{persona.character_name}'로서 응답해야 합니다.

[페르소나: '{persona.character_name}']
- 이름: {persona.character_name}
- 성격: {persona.persona}
- 설정: {persona.scenario}
- 대화 스타일: {persona.speaking_style}
- 제약 조건: {constraints}

[참고 맥락]
{rag_context}
"""
    
    BOS = "<s>"
    EOS = "</s>"
    B_INST = "[INST]"
    E_INST = "[/INST]"
    B_SYS = "<<SYS>>\n"
    E_SYS = "\n<</SYS>>\n\n"

    prompt_str = f"{BOS}{B_INST} {B_SYS}{system_content}{E_SYS}"
    
    dialogue_turns = persona.example_dialogue + prompt_input.recent_chat
    
    if dialogue_turns:
        first_turn = dialogue_turns[0]
        prompt_str += f"{first_turn.content} {E_INST}"
        
        if len(dialogue_turns) > 1 and dialogue_turns[1].role in ["assistant", "character"]:
            prompt_str += f" {dialogue_turns[1].content.strip()} {EOS}"
            
            remaining_turns = dialogue_turns[2:]
        else:
            prompt_str += f" {EOS}" # 턴을 닫음
            remaining_turns = dialogue_turns[1:]
    else:
        remaining_turns = []

    for i in range(0, len(remaining_turns), 2):
        user_turn = remaining_turns[i]
        prompt_str += f"{BOS}{B_INST} {user_turn.content} {E_INST}"
        
        if i + 1 < len(remaining_turns):
            assistant_turn = remaining_turns[i+1]
            prompt_str += f" {assistant_turn.content.strip()} {EOS}"
        else:
            prompt_str += f" {EOS}"

    prompt_str += f"{BOS}{B_INST} {prompt_input.user_message} {E_INST}"

    meta = {
        "chat_ctx_count": len(prompt_input.chat_context),
        "story_ctx_count": len(prompt_input.story_context),
        "recent_chat_count": len(prompt_input.recent_chat),
        "example_chat_count": len(persona.example_dialogue),
    }
    
    return PromptBuildOutput(prompt=prompt_str, meta=meta)