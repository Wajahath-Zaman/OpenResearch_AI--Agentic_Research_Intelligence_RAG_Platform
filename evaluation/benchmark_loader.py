import json
from pathlib import Path

from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_DATASET_PATH = (
    PROJECT_ROOT / "evaluation" / "dataset" / "research_benchmark.json"
)


class BenchmarkRecord(BaseModel):
    question: str = Field(min_length=1)
    reference: str = Field(min_length=1)
    reference_contexts: list[str] = Field(min_length=1)


class BenchmarkLoader:

    def __init__(self, dataset_path: Path = DEFAULT_DATASET_PATH):
        self.dataset_path = Path(dataset_path)

    def load(self) -> list[BenchmarkRecord]:

        if not self.dataset_path.exists():
            raise FileNotFoundError(
                f"Benchmark dataset not found: {self.dataset_path}"
            )

        try:
            with self.dataset_path.open("r", encoding="utf-8") as file:
                data = json.load(file)

        except json.JSONDecodeError as exc:
            raise ValueError("Invalid benchmark JSON file.") from exc

        if not isinstance(data, list):
            raise ValueError("Benchmark dataset must contain a JSON list.")

        return [
            BenchmarkRecord.model_validate(record)
            for record in data
        ]