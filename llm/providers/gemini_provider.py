from google import genai
from google.genai import types, errors

from config.settings import get_settings
from llm.providers.base import (
    LLMProvider,
    LLMProviderError,
    RetryableLLMProviderError
)

import httpx

class GeminiProvider(LLMProvider):

    def __init__(self, client=None):
        settings = get_settings()

        self.model = settings.gemini_model

        if client is not None:
            self.client = client
        else:
            if not settings.gemini_api_key:
                raise ValueError("Gemini API key is not configured.")

            self.client = genai.Client(
                api_key=settings.gemini_api_key,
                http_options=types.HttpOptions(
                    retry_options=types.HttpRetryOptions(
                        attempts=1
                    )
                )
            )

    def generate(
        self,
        messages: list[dict],
        max_completion_tokens: int = 1024
    ) -> str:

        if not isinstance(messages, list) or not messages:
            raise ValueError("Messages must be a non-empty list.")

        if (
            not isinstance(max_completion_tokens, int)
            or isinstance(max_completion_tokens, bool)
            or max_completion_tokens <= 0
        ):
            raise ValueError("Token limit must be a positive integer.")

        system_instructions = []
        contents = []

        for message in messages:

            if not isinstance(message, dict):
                raise ValueError("Each message must be a dictionary.")

            role = message.get("role")
            content = message.get("content")

            if not isinstance(role, str) or not isinstance(content, str):
                raise ValueError("Each message must contain string role and content.")

            if not content.strip():
                raise ValueError("Message content cannot be empty.")

            if role == "system":
                system_instructions.append(content)

            elif role in ("user", "assistant"):
                contents.append({
                    "role": "user" if role == "user" else "model",
                    "parts": [{"text": content}]
                })

            else:
                raise ValueError(f"Unsupported message role: {role}")

        if not contents:
            raise ValueError("At least one user or assistant message is required.")

        config = types.GenerateContentConfig(
            system_instruction="\n\n".join(system_instructions) or None,
            max_output_tokens=max_completion_tokens
        )

        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=contents,
                config=config
            )

            content = response.text

            if not isinstance(content, str) or not content.strip():
                raise LLMProviderError(
                    "Gemini returned an empty response."
                )

            return content.strip()

        except httpx.TransportError as exc:
            raise RetryableLLMProviderError(
                f"Gemini network failure: {exc}"
            ) from exc

        except errors.APIError as exc:

            status_code = getattr(exc, "code", None)

            if (
                status_code in {408, 429}
                or (
                    status_code is not None
                    and 500 <= status_code < 600
                )
            ):
                raise RetryableLLMProviderError(
                    f"Gemini temporary API failure ({status_code}): {exc}"
                ) from exc

            raise LLMProviderError(
                f"Gemini API request failed: {exc}"
            ) from exc