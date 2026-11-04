import pytest

from evaluation.retrieval_metrics import (
    precision_at_k,
    recall_at_k,
    mean_reciprocal_rank
)


# Precision@K tests

def test_precision_at_k():

    retrieved = ["A", "B", "C", "D", "E"]
    relevant = {"A", "C", "F"}

    assert precision_at_k(retrieved, relevant, 5) == pytest.approx(0.4)


def test_precision_at_k_with_all_relevant():

    retrieved = ["A", "B", "C"]
    relevant = {"A", "B", "C"}

    assert precision_at_k(retrieved, relevant, 3) == 1.0


def test_precision_at_k_with_no_relevant_results():

    retrieved = ["A", "B", "C"]
    relevant = {"D", "E"}

    assert precision_at_k(retrieved, relevant, 3) == 0.0


def test_precision_invalid_k():

    with pytest.raises(ValueError):

        precision_at_k(["A"], {"A"}, 0)


# Recall@K tests

def test_recall_at_k():

    retrieved = ["A", "B", "C", "D", "E"]
    relevant = {"A", "C", "F"}

    assert recall_at_k(retrieved, relevant, 5) == pytest.approx(2 / 3)


def test_recall_at_k_with_all_relevant():

    retrieved = ["A", "B", "C"]
    relevant = {"A", "B", "C"}

    assert recall_at_k(retrieved, relevant, 3) == 1.0


def test_recall_at_k_with_no_relevant_results():

    retrieved = ["A", "B", "C"]
    relevant = {"D", "E"}

    assert recall_at_k(retrieved, relevant, 3) == 0.0


def test_recall_invalid_k():

    with pytest.raises(ValueError):

        recall_at_k(["A"], {"A"}, -1)


# MRR tests

def test_mean_reciprocal_rank():

    retrieved_results = [
        ["A", "B", "C"],
        ["D", "E", "F"],
        ["G", "H", "I"]
    ]

    relevant_ids = [
        {"A"},
        {"E"},
        {"I"}
    ]

    expected = (1.0 + 0.5 + (1 / 3)) / 3

    assert mean_reciprocal_rank(
        retrieved_results,
        relevant_ids
    ) == pytest.approx(expected)


def test_mrr_with_no_relevant_results():

    retrieved_results = [
        ["A", "B"],
        ["C", "D"]
    ]

    relevant_ids = [
        {"X"},
        {"Y"}
    ]

    assert mean_reciprocal_rank(
        retrieved_results,
        relevant_ids
    ) == 0.0


def test_mrr_with_empty_results():

    retrieved_results = [
        [],
        ["A", "B"]
    ]

    relevant_ids = [
        {"X"},
        {"A"}
    ]

    assert mean_reciprocal_rank(
        retrieved_results,
        relevant_ids
    ) == pytest.approx(0.5)


def test_mrr_with_mismatched_lengths():

    with pytest.raises(ValueError):

        mean_reciprocal_rank(
            [["A"], ["B"]],
            [{"A"}]
        )