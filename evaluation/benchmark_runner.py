import asyncio
import json
from pathlib import Path

from agent.dependencies import create_answer_generator
from evaluation.benchmark_loader import BenchmarkLoader
from evaluation.ragas_evaluator import RAGASEvaluator


class BenchmarkRunner:

    def __init__(
        self,
        answer_generator=None,
        evaluator=None,
        benchmark_loader=None
    ):
        self.answer_generator = (
            answer_generator or create_answer_generator()
        )

        self.evaluator = evaluator or RAGASEvaluator()

        self.benchmark_loader = (
            benchmark_loader or BenchmarkLoader()
        )

    async def run(self):

        records = self.benchmark_loader.load()

        results = []

        for index, record in enumerate(records, start=1):

            print(
                f"\nEvaluating question {index}/{len(records)}:"
            )
            print(record.question)

            # Generate answer using actual RAG pipeline
            rag_result = self.answer_generator.generate_answer(
                question=record.question,
                n_results=5
            )

            retrieved_contexts = [
                chunk["text"]
                for chunk in rag_result["retrieved_chunks"]
            ]

            if not retrieved_contexts:
                print("No context retrieved. Skipping evaluation.")
                continue

            # Evaluate generated answer
            scores = await self.evaluator.evaluate_sample(
                user_input=record.question,
                response=rag_result["answer"],
                retrieved_contexts=retrieved_contexts,
                reference=record.reference
            )

            results.append({
                "question": record.question,
                "answer": rag_result["answer"],
                "reference": record.reference,
                "scores": scores
            })

            print("Scores:", scores)

        if not results:
            raise RuntimeError("No benchmark samples were evaluated.")

        metric_names = results[0]["scores"].keys()

        average_scores = {
            metric: sum(
                result["scores"][metric]
                for result in results
            ) / len(results)
            for metric in metric_names
        }

        output = {
            "total_samples": len(records),
            "evaluated_samples": len(results),
            "average_scores": average_scores,
            "results": results
        }

        output_path = (
            Path(__file__).resolve().parent
            / "benchmark_results.json"
        )

        output_path.write_text(
            json.dumps(output, indent=4),
            encoding="utf-8"
        )

        print("\nBenchmark Summary")
        print("=" * 40)
        print(json.dumps(average_scores, indent=4))
        print(f"\nResults saved to: {output_path}")

        return output


if __name__ == "__main__":
    runner = BenchmarkRunner()
    asyncio.run(runner.run())