import uuid
import enum

from datetime import datetime

from sqlalchemy import (
    String,
    Integer,
    Text,
    JSON,
    DateTime,
    Enum,
    Uuid,
    Index,
    func
)


from sqlalchemy.orm import Mapped, mapped_column

from database.base import Base


class DocumentStatus(str, enum.Enum):
    DISCOVERED = 'discovered'
    DOWNLOADED = 'downloaded'
    PROCESSED = 'processed'
    FAILED = 'failed'


class Document(Base):

    __tablename__ = "documents"

    __table_args__ = (
        Index("ix_documents_publication_year", "publication_year"),
        Index("ix_documents_status", "status")
    )

    # Primary Key
    id: Mapped[uuid.UUID] = mapped_column(
        Uuid, # UUIDs provide globally uniqe identifiers
        primary_key = True,
        default=uuid.uuid4
    )


    # Research Paper Information
    title: Mapped[str] = mapped_column(
        String(500),
        nullable=False
    )

    authors: Mapped[list] = mapped_column(
        JSON,
        nullable=False,
        default=list
    )

    abstract: Mapped[str | None] = mapped_column(
        Text,
        nullable=True
    )

    publication_year: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True
    )

    # External Identifiers
    doi: Mapped[str | None] = mapped_column(
        String(225),
        unique=True,
        nullable=True
    )

    openalex_id: Mapped[str | None] = mapped_column(
        String(500),
        unique=True,
        nullable=True
    )

    # Source And File Information
    source_url: Mapped[str | None] = mapped_column(
        String(1000),
        nullable=True
    )

    pdf_url: Mapped[str | None] = mapped_column(
        String(1000),
        nullable=True
    )

    file_path: Mapped[str | None] = mapped_column(
        String(1000),
        nullable=True
    )

    # Processing Information
    status: Mapped[DocumentStatus] = mapped_column(
        Enum(
            DocumentStatus,
            native_enum=False,
            values_callable=lambda enum_class: [
                item.value for item in enum_class
            ]
        ),
        nullable=False,
        default=DocumentStatus.DISCOVERED
    )

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=func.now(),
        onupdate=func.now(),
        nullable=False
    )


    def __repr__(self):
        return (
            f"Document(id={self.id}, "
            f"Title={self.title!r}, "
            f"status={self.status.value})"
        )