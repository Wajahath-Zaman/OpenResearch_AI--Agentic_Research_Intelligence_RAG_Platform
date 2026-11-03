from ingestion.research_discovery import OpenAlexClient
from ingestion.metadata_normalizer import normalize_work

from database.repositories import DocumentRepository

class ResearchDiscoveryService:

    def __init__(self, client: OpenAlexClient | None = None, repository: DocumentRepository | None = None):
        self.client = client or OpenAlexClient()
        self.repository = repository

    def discover(
        self,
        query: str,
        per_page: int = 10,
        page: int = 1
    ) -> dict:

        # Step 1: Retrieve raw research papers
        response = self.client.search_works(
            query=query,
            per_page=per_page,
            page=page
        )

        raw_works = response.get("results", [])

        normalized_works = []
        skipped_count = 0

        # Step 2: Normalize individual research papers
        for work in raw_works:

            try:
                normalized_work = normalize_work(work)
                normalized_works.append(normalized_work)

            except (ValueError, TypeError):
                skipped_count += 1

        # Step 3: Extract pagination metadata
        metadata = response.get("meta", {})

        total_count = metadata.get("count", len(raw_works))

        # Step 4: Return structured results
        return {
            "query": query.strip(),
            "total_count": total_count,
            "page": page,
            "per_page": per_page,
            "returned_count": len(normalized_works),
            "skipped_count": skipped_count,
            "results": normalized_works
        }


    def _find_existing_document(self, paper: dict):

        if paper.get("doi"):
            existing = self.repository.get_document_by_doi(
                paper['doi']
            )

            if existing:
                return existing

        if paper.get("openalex_id"):
            existing = self.repository.get_document_by_openalex_id(
                paper['openalex_id']
            )

            if existing:
                return existing


        return None


    def discover_and_store(
        self,
        query: str,
        per_page: int = 10,
        page: int = 1
    ) -> dict:

        if self.repository is None:
            raise RuntimeError(
                "DocumentRepository must be provided to store research papers."
            )

        # Step 1: Discover and normalize papers
        discovery_result = self.discover(
            query=query,
            per_page=per_page,
            page=page
        )

        saved_documents = []
        duplicate_documents = []
        unidentifiable_documents = []

        seen_dois = set()
        seen_openalex_ids = set()

        # Step 2: Process each paper exactly once
        for paper in discovery_result["results"]:

            doi = paper.get("doi")
            openalex_id = paper.get("openalex_id")

            has_identifier = bool(doi or openalex_id)

            # Check duplicates within the incoming batch
            duplicate_in_batch = (
                (doi and doi in seen_dois)
                or
                (openalex_id and openalex_id in seen_openalex_ids)
            )

            if duplicate_in_batch:
                duplicate_documents.append(paper)
                continue

            # Check existing database records
            existing_document = self._find_existing_document(paper)

            if existing_document:
                duplicate_documents.append(existing_document)

            else:
                new_document = self.repository.create_document(**paper)

                saved_documents.append(new_document)

                if not has_identifier:
                    unidentifiable_documents.append(new_document)

            # Track identifiers
            if doi:
                seen_dois.add(doi)

            if openalex_id:
                seen_openalex_ids.add(openalex_id)

        # Step 3: Return processing statistics
        return {
            **discovery_result,
            "saved_count": len(saved_documents),
            "duplicate_count": len(duplicate_documents),
            "unidentifiable_count": len(unidentifiable_documents),
            "saved_documents": saved_documents,
            "duplicate_documents": duplicate_documents,
            "unidentifiable_documents": unidentifiable_documents
        }

        return {
            **discovery_result,
            "saved_count": len(saved_documents),
            "duplicate_count": len(duplicate_documents),
            "unidentifiable_count": len(unidentifiable_documents),
            "saved_documents": saved_documents,
            "duplicate_documents": duplicate_documents,
            "unidentifiable_documents": unidentifiable_documents
        }