import asyncio
import json

from evaluation.ragas_evaluator import RAGASEvaluator


async def main():
    evaluator = RAGASEvaluator()

    result = await evaluator.evaluate_sample(
        user_input=(
            "Which transformer architecture achieved the highest "
            "classification accuracy in the study?"
        ),
        response=(
            "Swin-Base achieved the highest classification accuracy "
            "of 98.80%."
        ),
        retrieved_contexts=[
            (
                "The experimental results showed that the Swin-Base "
                "architecture achieved an accuracy of 98.80%, outperforming "
                "the other evaluated transformer architectures."
            )
        ],
        reference=(
            "Swin-Base achieved the highest classification accuracy "
            "with 98.80%."
        )
    )

    print("\nRAGAS Evaluation Results")
    print("=" * 40)
    print(json.dumps(result, indent=4))


if __name__ == "__main__":
    asyncio.run(main())