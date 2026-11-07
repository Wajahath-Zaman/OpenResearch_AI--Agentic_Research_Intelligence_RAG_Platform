from ragas.metrics.collections import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecisionWithReference,
    ContextRecall
)

from evaluation.ragas_llm import create_ragas_llm
from evaluation.ragas_embeddings import RagasEmbeddingAdapter


class RAGASEvaluator:

    def __init__(self, llm=None, embeddings=None):

        self.llm = llm or create_ragas_llm()
        self.embeddings = embeddings or RagasEmbeddingAdapter()

        self.metrics = {
            "faithfulness": Faithfulness(llm=self.llm),

            "answer_relevancy": AnswerRelevancy(
                llm=self.llm,
                embeddings=self.embeddings
            ),

            "context_precision": ContextPrecisionWithReference(
                llm=self.llm
            ),

            "context_recall": ContextRecall(
                llm=self.llm
            )
        }

    async def evaluate_sample(
        self,
        user_input: str,
        response: str,
        retrieved_contexts: list[str],
        reference: str
    ) -> dict:

        if not isinstance(user_input, str) or not user_input.strip():
            raise ValueError("User input must be a non-empty string.")

        if not isinstance(response, str) or not response.strip():
            raise ValueError("Response must be a non-empty string.")

        if not isinstance(retrieved_contexts, list) or not retrieved_contexts:
            raise ValueError("Retrieved contexts must be a non-empty list.")

        if not all(
            isinstance(context, str) and context.strip()
            for context in retrieved_contexts
        ):
            raise ValueError("All retrieved contexts must be non-empty strings.")

        if not isinstance(reference, str) or not reference.strip():
            raise ValueError("Reference must be a non-empty string.")

        results = {}

        # 1. Faithfulness
        result = await self.metrics["faithfulness"].ascore(
            user_input=user_input,
            response=response,
            retrieved_contexts=retrieved_contexts
        )

        results["faithfulness"] = result.value

        # 2. Answer Relevancy
        result = await self.metrics["answer_relevancy"].ascore(
            user_input=user_input,
            response=response
        )

        results["answer_relevancy"] = result.value

        # 3. Context Precision
        result = await self.metrics["context_precision"].ascore(
            user_input=user_input,
            reference=reference,
            retrieved_contexts=retrieved_contexts
        )

        results["context_precision"] = result.value

        # 4. Context Recall
        result = await self.metrics["context_recall"].ascore(
            user_input=user_input,
            reference=reference,
            retrieved_contexts=retrieved_contexts
        )

        results["context_recall"] = result.value

        return results