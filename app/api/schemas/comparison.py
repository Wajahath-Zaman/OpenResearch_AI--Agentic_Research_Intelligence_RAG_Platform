from pydantic import BaseModel, Field


class PaperComparisonRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    document_ids: list[str] = Field(min_length=2)


class PaperComparisonResponse(BaseModel):
    comparison: str
    comparison_results: list[dict]