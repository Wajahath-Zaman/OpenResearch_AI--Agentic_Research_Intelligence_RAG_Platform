from llm.providers.groq_provider import GroqProvider


def main():

    provider = GroqProvider()

    messages = [
        {
            "role": "system",
            "content": "You are a helpful AI research assistant."
        },
        {
            "role": "user",
            "content": "Explain RAG in three sentences."
        }
    ]

    response = provider.generate(
        messages=messages,
        max_completion_tokens=256
    )

    print("\nGroq Response:\n")
    print(response)


if __name__ == "__main__":
    main()