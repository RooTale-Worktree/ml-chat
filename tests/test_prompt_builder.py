from src.core.schemas import Persona, DialogueTurn, RAGChunk, PromptBuildInput
from src.core.prompt_builder import build_prompt


def test_prompt_contains_sections():
    persona = Persona(
        name="Aria",
        persona="An adventurer.",
        scenario="In a desert.",
        speaking_style="Casual",
        constraints=["Stay positive"],
        example_dialogue=[DialogueTurn(role="character", text="안녕!"), DialogueTurn(role="user", text="잘 지냈어?")]
    )
    chat_ctx = [RAGChunk(id="c1", source="chat", text="user: hi", score=None)]
    story_ctx = [RAGChunk(id="s1", source="story", text="사막은 뜨거웠다", score=None)]
    hist = [DialogueTurn(role="user", text="어제 뭐 했어?"), DialogueTurn(role="character", text="준비했지.")]

    inp = PromptBuildInput(
        persona=persona,
        chat_context=chat_ctx,
        story_context=story_ctx,
        history=hist,
        user_message="계획이 뭐야?"
    )
    out = build_prompt(inp)
    assert "<<SYSTEM>>" in out.prompt
    assert "<<CONTEXT:CHAT>>" in out.prompt
    assert "<<CONTEXT:STORY>>" in out.prompt
    assert "<<CONVERSATION>>" in out.prompt
