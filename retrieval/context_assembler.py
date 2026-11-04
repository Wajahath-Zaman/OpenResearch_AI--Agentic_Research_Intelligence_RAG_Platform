class ContextAssembler:

    def assemble(self, retrieved_chunks: list[dict]) -> str:
        """
        Assemble retrieved document chunks into an LLM-ready context.
        """

        if not isinstance(retrieved_chunks, list):
            raise TypeError("Retrieved chunks must be a list.")

        if not retrieved_chunks:
            return ""

        context_parts = []

        for index, chunk in enumerate(retrieved_chunks, start=1):

            if not isinstance(chunk, dict):
                raise TypeError("Each retrieved chunk must be a dictionary.")

            text = chunk.get("text")

            if not isinstance(text, str) or not text.strip():
                raise ValueError("Retrieved chunk must contain valid text.")

            metadata = chunk.get("metadata", {})

            if not isinstance(metadata, dict):
                raise TypeError("Chunk metadata must be a dictionary.")

            title = metadata.get("title", "Unknown Document")
            page_number = metadata.get("page_number", "Unknown")

            passage = (
                f"--- Source {index} ---\n"
                f"Document: {title}\n"
                f"Page: {page_number}\n"
                f"Content:\n{text.strip()}"
            )

            context_parts.append(passage)

        return "\n\n".join(context_parts)