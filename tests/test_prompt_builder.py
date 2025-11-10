import uuid

from src.prompt.get_prompt import get_prompt
from src.schemas.rag import PromptBuildInput, RAGChunk
from src.schemas.request import DialogueTurn, Persona


def _persona() -> Persona:
    return Persona(
        character_id=uuid.uuid4(),
        character_name="Aria",
        persona="An adventurer exploring forgotten ruins.",
        scenario="Searching for an oasis in a desert.",
        speaking_style="Casual and optimistic",
        constraints=["Stay positive"],
        example_dialogue=[
            DialogueTurn(role="character", content="안녕! 오늘도 모험이야."),
            DialogueTurn(role="user", content="위험하지 않아?"),
        ],
    )


def _prompt_input() -> PromptBuildInput:
    persona = _persona()
    chat_ctx = [RAGChunk(id="c1", source="chat", text="User: 안녕!", score=None)]
    story_ctx = [RAGChunk(id="s1", source="story", text="사막은 뜨거웠다", score=None)]
    recent = [
        DialogueTurn(role="user", content="어제 뭐 했어?"),
        DialogueTurn(role="character", content="준비했지."),
    ]
    return PromptBuildInput(
        persona=persona,
        chat_context=chat_ctx,
        story_context=story_ctx,
        recent_chat=recent,
        user_message="오늘 계획이 뭐야?",
    )


def test_pygmalion_prompt_contains_sections():
    out = get_prompt("pygmalion-6b", _prompt_input())
    assert "<<SYSTEM>>" in out.prompt
    assert "<<CONTEXT:CHAT>>" in out.prompt
    assert "<<CONTEXT:STORY>>" in out.prompt
    assert "<<CONVERSATION>>" in out.prompt


def test_gpt_oss_prompt_uses_instruction_blocks():
    out = get_prompt("gpt-oss-20b", _prompt_input())
    assert "### Instruction" in out.prompt
    assert "### Conversation" in out.prompt
