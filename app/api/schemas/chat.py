from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    document_ids: list[str] = Field(default_factory=list)


class ChatResponse(BaseModel):
    query: str
    route: str
    answer: str
    citations: list[dict] = Field(default_factory=list)
    research_results: list[dict] = Field(default_factory=list)
    comparison_results: list[dict] = Field(default_factory=list)
    error: str | None = None