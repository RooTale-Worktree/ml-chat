"""
Prompt builder that renders persona + RAG context using Harmony format for gpt-oss.
"""
from __future__ import annotations
from datetime import datetime
from typing import List, Dict
from openai_harmony import (
    Author,
    Conversation,
    DeveloperContent,
    HarmonyEncodingName,
    Message,
    Role,
    SystemContent,
    load_harmony_encoding,
    ReasoningEffort
)


def _build_persona_context(persona: Dict, user_name: str) -> str:
    """
    Build persona context string from Persona dict.
    """
    character_name = persona.get("character_name", "Character")
    persona_desc = persona.get("persona", "")
    scenario = persona.get("scenario", "")
    speaking_style = persona.get("speaking_style", [])
    constraints = persona.get("constraints", [])
    example_dialogue = persona.get("example_dialogue", [])

    speaking_style_str = "\n".join([f"{i+1}. {c}" for i, c in enumerate(speaking_style)])
    constraints_str = "\n".join([f"{i+1}. {c}" for i, c in enumerate(constraints)])
    example_dialogue_str = ""
    if example_dialogue:
        example_dialogue_str = "[대화 예시]\n" + "\n".join(
            [f"{turn['role'].upper()}: {turn['content']}" for turn in example_dialogue]
        )

    # Construct persona context
    response_format_instruction = """
[응답 형식 (필수)]
당신은 반드시 아래의 json 형식으로만 응답해야 합니다. 마크다운이나 기타 설명을 붙이지 마세요.
{
    "narrative": "캐릭터의 행동, 표정, 속마음 또는 상황 묘사 (3인칭 시점)",
    "character": "캐릭터가 실제로 말하는 대사"
}
"""

    persona_context = f"""
당신은 '{character_name}'입니다. 당신은 '{user_name}'과 대화 중입니다.
아래의 페르소나와 지침을 바탕으로 완벽하게 연기하세요.

[핵심 정체성]
{persona_desc}

[대화 문맥]
{scenario}

[말투 및 스타일 (반드시 준수)]
{speaking_style_str}

[제약 사항 (반드시 준수)]
{constraints_str}

{response_format_instruction}

{example_dialogue_str}
""".strip()
    return persona_context


def build_prompt(prompt_input: Dict) -> List[Dict]:

    persona = prompt_input.get("persona", None)
    user_name = prompt_input.get("user_name", "User")
    chat_context = prompt_input.get("chat_context", [])
    story_context = prompt_input.get("story_context", [])
    user_message = prompt_input.get("user_message", "")
    reasoning_effort = prompt_input.get("reasoning_effort", "low")

    if persona is None:
        raise ValueError("Persona information is required to build prompt.")
    
    # Load Harmony encoding
    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

    # Define persona content
    persona_content = (
        DeveloperContent.new()
        .with_instructions(_build_persona_context(persona, user_name))
    )

    # Define system content
    system_config = SystemContent.new()
    if reasoning_effort == "high":
        system_config = system_config.with_reasoning_effort(ReasoningEffort.HIGH)
    elif reasoning_effort == "medium":
        system_config = system_config.with_reasoning_effort(ReasoningEffort.MEDIUM)
    else:
        system_config = system_config.with_reasoning_effort(ReasoningEffort.LOW)
    system_config.conversation_start_date = datetime.today().strftime("%Y-%m-%d")
    system_messages = Message.from_role_and_content(Role.SYSTEM, system_config)

    # Define chat history content
    history_messages = []
    for msg in chat_context:
        role = msg.get("role", "user").lower()
        if role == "user":
            history_messages.append(
                Message.from_role_and_content(Role.USER, msg.get("content", ""))
            )
        elif role == "assistant":
            history_messages.append(
                Message.from_role_and_content(Role.ASSISTANT, msg.get("content", ""))
            )

    # Define story context content
    story_message_list = []
    if story_context:
        story_text = "[스토리 관련 정보]\n" + "\n".join(
            [f"- {story['text']}" for story in story_context]
        )
        story_content = DeveloperContent.new().with_instructions(story_text)
        story_message_list.append(
            Message.from_role_and_content(Role.DEVELOPER, story_content)
        )

    # Current user message
    current_user_input = Message.from_role_and_content(Role.USER, user_message)

    # Build conversation
    conversation = Conversation.from_messages(
        [system_messages] +
        [Message.from_role_and_content(Role.DEVELOPER, persona_content)] +
        story_message_list +
        history_messages +
        [current_user_input]
    )

    # Tokenize for completion
    tokens = encoding.render_conversation_for_completion(conversation, Role.ASSISTANT)
    final_prompt = [{"prompt_token_ids": tokens}]

    return final_prompt


__all__ = ["build_prompt"]
