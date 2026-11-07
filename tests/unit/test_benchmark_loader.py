import json
import pytest

from pydantic import ValidationError

from evaluation.benchmark_loader import (
    BenchmarkLoader,
    BenchmarkRecord
)


def test_load_default_benchmark_dataset():
    loader = BenchmarkLoader()

    records = loader.load()

    assert len(records) == 8
    assert all(isinstance(record, BenchmarkRecord) for record in records)


def test_load_custom_dataset(tmp_path):
    dataset = [
        {
            "question": "What is a transformer?",
            "reference": "A transformer uses attention mechanisms.",
            "reference_contexts": ["Transformers use self-attention."]
        }
    ]

    path = tmp_path / "benchmark.json"
    path.write_text(json.dumps(dataset), encoding="utf-8")

    loader = BenchmarkLoader(path)
    records = loader.load()

    assert len(records) == 1
    assert records[0].question == "What is a transformer?"


def test_missing_dataset_raises_error(tmp_path):
    loader = BenchmarkLoader(tmp_path / "missing.json")

    with pytest.raises(FileNotFoundError):
        loader.load()


def test_invalid_json_raises_error(tmp_path):
    path = tmp_path / "invalid.json"
    path.write_text("{invalid json", encoding="utf-8")

    loader = BenchmarkLoader(path)

    with pytest.raises(ValueError, match="Invalid benchmark JSON"):
        loader.load()


def test_non_list_dataset_raises_error(tmp_path):
    path = tmp_path / "benchmark.json"
    path.write_text('{"question": "What is AI?"}', encoding="utf-8")

    loader = BenchmarkLoader(path)

    with pytest.raises(ValueError, match="JSON list"):
        loader.load()


def test_invalid_record_raises_validation_error(tmp_path):
    dataset = [
        {
            "question": "",
            "reference": "Expected answer",
            "reference_contexts": ["Supporting context"]
        }
    ]

    path = tmp_path / "benchmark.json"
    path.write_text(json.dumps(dataset), encoding="utf-8")

    loader = BenchmarkLoader(path)

    with pytest.raises(ValidationError):
        loader.load()