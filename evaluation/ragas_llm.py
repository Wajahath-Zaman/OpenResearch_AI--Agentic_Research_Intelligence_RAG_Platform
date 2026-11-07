from openai import AsyncOpenAI
from ragas.llms import llm_factory

from config.settings import get_settings


def create_ragas_llm():

    settings = get_settings()

    if not settings.groq_api_key:
        raise ValueError("Groq API key is not configured.")

    client = AsyncOpenAI(
        api_key=settings.groq_api_key,
        base_url="https://api.groq.com/openai/v1"
    )

    evaluator_llm = llm_factory(
        model=settings.groq_model,
        provider="openai",
        client=client,
        max_tokens=4096
    )

    return evaluator_llm