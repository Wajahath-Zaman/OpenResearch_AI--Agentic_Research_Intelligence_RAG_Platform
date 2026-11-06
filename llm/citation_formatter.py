from retrieval.source_tracker import SourceTracker


class CitationFormatter:

    def __init__(self, source_tracker: SourceTracker | None = None):
        self.source_tracker = source_tracker or SourceTracker()

    def format_sources(self, retrieved_chunks: list[dict]) -> list[dict]:

        sources = self.source_tracker.extract_sources(retrieved_chunks)

        formatted_sources = []

        for index, source in enumerate(sources, start=1):
            formatted_sources.append({
                "citation_id": index,
                "document_id": source["document_id"],
                "title": source.get("title"),
                "page_number": source["page_number"],
                "source_path": source.get("source_path")
            })

        return formatted_sources