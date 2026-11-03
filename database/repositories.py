import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from database.models import Document, DocumentStatus


class DocumentRepository:


    def __init__(self, db: Session):
        self.db = db

    # Creating a document
    def create_document(self, **document_data) -> Document:

        document = Document(**document_data)

        self.db.add(document)
        self._commit()
        self.db.refresh(document)

        return document


    # Retrive Document by UUID
    def get_document_by_id(
            self,
            document_id: uuid.UUID
    ) -> Document | None:

        statement = select(Document).where(
            Document.id == document_id
        )

        return self.db.scalar(statement)

    
    def get_document_by_doi(self, doi:str) -> Document | None:

        statement = select(Document).where(
            Document.doi == doi
        )

        return self.db.scalar(statement)



    # Retrieve all documents
    def get_all_documents(
            self,
            skip: int = 0,
            limit: int = 100
    ) -> list[Document]:

        statement = (
            select(Document)
            .offset(skip)
            .limit(limit)
            .order_by(Document.created_at.desc())
        )

        return list(self.db.scalars(statement).all())
    
    def get_document_by_status(
        self,
        status: DocumentStatus
    ) -> list[Document]:

        statement = select(Document). where(
            Document.status == status
        )

        return list(self.db.scalars(statement).all())


    # Update Document processing status
    def update_document_status(
            self,
            document_id: uuid.UUID,
            status: DocumentStatus
    ) -> Document | None:

        document = self.get_document_by_id(document_id)

        if document is None:
            return None

        document.status = status

        self._commit()
        self.db.refresh(document)

        return document


    # Delete document
    def delete_document(
            self,
            document_id: uuid.UUID
    ) -> bool:

        document = self.get_document_by_id(document_id)

        if document is None:
            return False

        self.db.delete(document)
        self._commit()

        return True


    def _commit(self):
        try:
            self.db.commit()

        except Exception:
            self.db.rollback()
            raise