from uuid import UUID

from database.models import DocumentStatus
from ingestion.pdf_loader import PDFLoader
from ingestion.preprocessing import TextPreprocessor
from ingestion.chunking import TextChunker
from ingestion.embedding_generator import EmbeddingGenerator
from retrieval.vector_store import ChromaVectorStore


class DocumentIngestionService:

    def __init__(
        self,
        repository,
        pdf_loader=None,
        preprocessor=None,
        chunker=None,
        embedding_generator=None,
        vector_store=None
    ):

        self.repository = repository

        self.pdf_loader = pdf_loader or PDFLoader()
        self.preprocessor = preprocessor or TextPreprocessor()
        self.chunker = chunker or TextChunker()
        self.embedding_generator = (
            embedding_generator or EmbeddingGenerator()
        )
        self.vector_store = vector_store or ChromaVectorStore()

    def ingest_document(self, document_id: UUID) -> dict:
        """
        Execute the complete document ingestion workflow.
        """

        document = self.repository.get_document_by_id(document_id)

        if document is None:
            raise ValueError("Document not found in database.")

        try:

            if not document.file_path:
                raise ValueError("Document does not have a PDF file path.")

            # Step 1: Extract PDF content
            extracted_document = self.pdf_loader.load(
                document.file_path
            )

            # Step 2: Preprocess extracted text
            processed_document = self.preprocessor.preprocess_document(
                extracted_document
            )

            # Step 3: Generate text chunks
            chunks = self.chunker.chunk_document(
                processed_document
            )

            if not chunks:
                raise ValueError(
                    "No extractable text found in the document."
                )

            # Step 4: Generate embeddings
            embedded_chunks = self.embedding_generator.embed_chunks(
                chunks
            )

            # Step 5: Store chunks and embeddings
            stored_count = self.vector_store.store_chunks(
                document_id=str(document.id),
                chunks=embedded_chunks,
                source_path=extracted_document["source_path"],
                title=(
                    extracted_document["metadata"]["title"]
                    or document.title
                )
            )

            # Step 6: Update document status
            self.repository.update_document_status(
                document.id,
                DocumentStatus.PROCESSED
            )

            return {
                "document_id": str(document.id),
                "title": document.title,
                "total_chunks": stored_count,
                "status": DocumentStatus.PROCESSED.value
            }

        except Exception:

            self.repository.update_document_status(
                document.id,
                DocumentStatus.FAILED
            )

            raise