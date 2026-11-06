from llm.prompts.templates import PromptTemplates


class RAGPromptBuilder:

    @staticmethod
    def build_messages(context: str, question: str) -> list[dict]:

        if not isinstance(context, str) or not context.strip():
            raise ValueError("Context must be a non-empty string.")

        if not isinstance(question, str) or not question.strip():
            raise ValueError("Question must be a non-empty string.")

        system_prompt = PromptTemplates.research_assistant_system_prompt()

        user_prompt = f"""
        Answer the user's question using the research context provided below.

        Instructions:
        - Base your answer on the provided context.
        - Do not invent facts or research findings.
        - If the context does not contain sufficient information,
          explicitly acknowledge the limitation.
        - Distinguish information supported by the context from
          any uncertainty.
        - Provide clear and structured answers.

        Research Context:
        {context}

        User Question:
        {question}
        """

        return [
            {
                "role": "system",
                "content": system_prompt.strip()
            },
            {
                "role": "user",
                "content": user_prompt.strip()
            }
        ]