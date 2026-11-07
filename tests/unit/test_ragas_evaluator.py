import pytest
from unittest.mock import AsyncMock, MagicMock

from evaluation.ragas_evaluator import RAGASEvaluator


@pytest.fixture
def evaluator():
    metrics = {
        "faithfulness": MagicMock(),
        "answer_relevancy": MagicMock(),
        "context_precision": MagicMock(),
        "context_recall": MagicMock()
    }

    for metric in metrics.values():
        metric.ascore = AsyncMock(
            return_value=MagicMock(value=0.90)
        )

    instance = RAGASEvaluator.__new__(RAGASEvaluator)
    instance.llm = MagicMock()
    instance.embeddings = MagicMock()
    instance.metrics = metrics

    return instance


@pytest.mark.anyio
async def test_evaluate_sample_returns_all_metrics(evaluator):

    result = await evaluator.evaluate_sample(
        user_input="What is a CNN?",
        response="CNNs are neural networks used for image processing.",
        retrieved_contexts=["CNNs extract spatial features from images."],
        reference="CNNs are neural networks commonly used for image tasks."
    )

    assert set(result.keys()) == {
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall"
    }

    assert all(score == 0.90 for score in result.values())


@pytest.mark.anyio
async def test_evaluate_sample_calls_all_metrics(evaluator):

    await evaluator.evaluate_sample(
        user_input="What is a CNN?",
        response="CNNs process images.",
        retrieved_contexts=["CNNs process visual information."],
        reference="CNNs are used in image processing."
    )

    for metric in evaluator.metrics.values():
        metric.ascore.assert_awaited_once()


@pytest.mark.anyio
async def test_rejects_empty_user_input(evaluator):

    with pytest.raises(ValueError):
        await evaluator.evaluate_sample(
            user_input="",
            response="Answer",
            retrieved_contexts=["Context"],
            reference="Reference"
        )


@pytest.mark.anyio
async def test_rejects_empty_response(evaluator):

    with pytest.raises(ValueError):
        await evaluator.evaluate_sample(
            user_input="Question",
            response="",
            retrieved_contexts=["Context"],
            reference="Reference"
        )


@pytest.mark.anyio
async def test_rejects_empty_contexts(evaluator):

    with pytest.raises(ValueError):
        await evaluator.evaluate_sample(
            user_input="Question",
            response="Answer",
            retrieved_contexts=[],
            reference="Reference"
        )


@pytest.mark.anyio
async def test_rejects_invalid_context_entries(evaluator):

    with pytest.raises(ValueError):
        await evaluator.evaluate_sample(
            user_input="Question",
            response="Answer",
            retrieved_contexts=["Valid context", ""],
            reference="Reference"
        )


@pytest.mark.anyio
async def test_rejects_empty_reference(evaluator):

    with pytest.raises(ValueError):
        await evaluator.evaluate_sample(
            user_input="Question",
            response="Answer",
            retrieved_contexts=["Context"],
            reference=""
        )