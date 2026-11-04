from pydantic import BaseModel, Field, model_validator


class ExtractedPage(BaseModel):
    page_number: int = Field(gt=0)
    text: str


class ExtractedDocumentMetadata(BaseModel):
    title: str | None = None
    author: str | None = None
    total_pages: int = Field(ge=1)


class ExtractedDocument(BaseModel):
    source_path: str
    metadata: ExtractedDocumentMetadata
    pages: list[ExtractedPage]

    @model_validator(mode="after")
    def validate_page_count(self):
        if len(self.pages) != self.metadata.total_pages:
            raise ValueError(
                "Metadata page count does not match extracted pages."
            )

        return self