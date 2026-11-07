import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from evaluation.benchmark_loader import BenchmarkRecord
from evaluation.benchmark_runner import BenchmarkRunner


@pytest.fixture
def benchmark_records():
    return [
        BenchmarkRecord(
            question="What is a transformer?",
            reference="A transformer uses attention mechanisms.",
            reference_contexts=["Transformers use self-attention."]
        ),
        BenchmarkRecord(
            question="What is Swin Transformer?",
            reference="Swin uses shifted-window attention.",
            reference_contexts=["Swin uses shifted local windows."]
        )
    ]


@pytest.fixture
def runner(benchmark_records):
    answer_generator = MagicMock()
    answer_generator.generate_answer.return_value = {
        "answer": "A transformer uses attention mechanisms.",
        "retrieved_chunks": [
            {"text": "Transformers use self-attention."}
        ]
    }

    evaluator = MagicMock()
    evaluator.evaluate_sample = AsyncMock(
        return_value={
            "faithfulness": 0.9,
            "answer_relevancy": 0.8,
            "context_precision": 0.85,
            "context_recall": 0.95
        }
    )

    loader = MagicMock()
    loader.load.return_value = benchmark_records

    return BenchmarkRunner(
        answer_generator=answer_generator,
        evaluator=evaluator,
        benchmark_loader=loader
    )


def test_benchmark_runner_evaluates_all_records(runner, monkeypatch):
    monkeypatch.setattr(
        "evaluation.benchmark_runner.Path.write_text",
        lambda *args, **kwargs: None
    )

    result = asyncio.run(runner.run())

    assert result["total_samples"] == 2
    assert result["evaluated_samples"] == 2
    assert len(result["results"]) == 2

    runner.evaluator.evaluate_sample.assert_awaited()


def test_benchmark_runner_calculates_average_scores(runner, monkeypatch):
    monkeypatch.setattr(
        "evaluation.benchmark_runner.Path.write_text",
        lambda *args, **kwargs: None
    )

    result = asyncio.run(runner.run())

    assert result["average_scores"]["faithfulness"] == pytest.approx(0.9)
    assert result["average_scores"]["answer_relevancy"] == pytest.approx(0.8)
    assert result["average_scores"]["context_precision"] == pytest.approx(0.85)
    assert result["average_scores"]["context_recall"] == pytest.approx(0.95)


def test_benchmark_runner_skips_samples_without_context(runner, monkeypatch):
    runner.answer_generator.generate_answer.return_value = {
        "answer": "I don't know.",
        "retrieved_chunks": []
    }

    with pytest.raises(RuntimeError, match="No benchmark samples"):
        asyncio.run(runner.run())

    runner.evaluator.evaluate_sample.assert_not_awaited()


def test_benchmark_runner_rejects_empty_benchmark(runner):
    runner.benchmark_loader.load.return_value = []

    with pytest.raises(RuntimeError, match="No benchmark samples"):
        asyncio.run(runner.run())