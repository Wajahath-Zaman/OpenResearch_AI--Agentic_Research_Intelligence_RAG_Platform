from groq import (
    Groq,
    APIError,
    APIConnectionError,
    APITimeoutError
)

from config.settings import get_settings
from llm.providers.base import (
    LLMProvider,
    LLMProviderError,
    RetryableLLMProviderError
)


class GroqProvider(LLMProvider):

    def __init__(self, client=None):

        settings = get_settings()

        self.model = settings.groq_model

        if client is not None:
            self.client = client


        else:
            if not settings.groq_api_key:
                raise ValueError("Grop API key is not configured.")

            self.client = Groq(
                api_key=settings.groq_api_key,
                max_retries=0
            )


    def generate(
            self,
            messages: list[dict],
            max_completion_tokens:int = 1024
    ) -> str:
        """
        Generate a response using the groq chat completions API.
        """

        if not isinstance(messages, list):
            raise TypeError("Messages Must be a list.")

        if not messages:
            raise ValueError("Messages Cannot be empty.")

        if (
            not isinstance(max_completion_tokens, int)
            or isinstance(max_completion_tokens, bool)
        ):
            raise TypeError("Maximum completion tokens must be an integer.")

        if max_completion_tokens <=0:
            raise ValueError("Maximum Completion tokens must be positive.")

        for message in messages:

            if not isinstance(message, dict):
                raise TypeError("Each Message must be a dictionary.")

            if not isinstance(message.get("role"), str):
                raise ValueError("Each Message must contain a valid role.")

            if not isinstance(message.get("content"), str):
                raise ValueError("Each Message must contain string content.")

        try:

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_completion_tokens=max_completion_tokens
            )

            content = response.choices[0].message.content

            if not isinstance(content, str) or not content.strip():
                raise LLMProviderError(
                    "Groq Returned an Empty response."
                )

            return content.strip()

        except (APIConnectionError, APITimeoutError) as exc:
            raise RetryableLLMProviderError(
                f"Groq connection failure: {exc}"
            ) from exc

        except APIError as exc:

            status_code = getattr(exc, "status_code", None)

            if (
                status_code in {408, 409, 429}
                or (status_code is not None and status_code >= 500)
            ):
                raise RetryableLLMProviderError(
                    f"Groq temporary API failure ({status_code}): {exc}"
                ) from exc

            raise LLMProviderError(
                f"Groq API request failed: {exc}"
            ) from exc