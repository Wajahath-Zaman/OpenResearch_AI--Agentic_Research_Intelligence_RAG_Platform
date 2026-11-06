from llm.prompts.templates import PromptTemplates


def test_research_assistant_prompt_returns_string():
    prompt = PromptTemplates.research_assistant_system_prompt()

    assert isinstance(prompt, str)


def test_research_assistant_prompt_is_not_empty():
    prompt = PromptTemplates.research_assistant_system_prompt()

    assert prompt.strip()


def test_research_assistant_prompt_contains_role():
    prompt = PromptTemplates.research_assistant_system_prompt()

    assert "AI research assistant" in prompt


def test_research_assistant_prompt_prevents_fabrication():
    prompt = PromptTemplates.research_assistant_system_prompt()

    assert "Do not fabricate" in prompt