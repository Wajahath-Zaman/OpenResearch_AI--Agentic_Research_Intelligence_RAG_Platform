def reconstruct_abstract(inverted_index: dict | None) -> str | None:

    if not inverted_index:
        return None


    positioned_words = {}

    for word, positions in inverted_index.items():

        for position in positions:

            if (
                isinstance(position, int)
                and not isinstance(position, bool)
                and position >= 0 
            ): 
                positioned_words[position] = word


    if not positioned_words:
        return None

    ordered_words = [
        positioned_words[position]
        for position in sorted(positioned_words)
    ]

    return " ".join(ordered_words)


def normalize_doi(doi: str | None) -> str | None:

    if not doi:
        return None

    doi = doi.strip()

    prefixes = [
        "https://doi.org/",
        "http://doi.org/",
        "https://dx.doi.org/",
        "http://dx.doi.org/"
    ]

    for prefix in prefixes:

        if doi.lower().startswith(prefix):
            doi = doi[len(prefix):]
            break

    return doi or None


def normalize_openalex_id(
        openalex_id: str | None
) -> str | None :

    if not openalex_id:
        return None

    return openalex_id.rstrip("/").split("/")[-1]


def normalize_work(work: dict) -> dict:

    if not isinstance(work, dict):
        raise ValueError("OpenAlex work must be a dictionary.")

    # Extract Title
    title = work.get("title")

    if not isinstance(title, str) or not title.strip():
        raise ValueError("Research Paper Must have a valid Title.")

    title = title.strip()

    # Extract Authors
    authors = []

    authorships = work.get("authorships") or []

    if isinstance(authorships, list):

        for authorship in authorships:

            if not isinstance(authorship, dict):
                continue

            author = authorship.get("author") or {}

            if not isinstance(author, dict):
                continue

            author_name = author.get("display_name")

            if isinstance(author_name, str) and author_name.strip():
                authors.append(author_name.strip())

    # Reconstruct Abstract
    abstract = reconstruct_abstract(
        work.get("abstract_inverted_index")
    )


    # Normalize Identifiers
    openalex_id = normalize_openalex_id(
        work.get("id")
    )

    doi = normalize_doi(
        work.get("doi")
    )


    # Extract source and PDF URLs
    best_location = work.get("best_oa_location") or {}
    primary_location = work.get("primary_location") or {}

    source_url = None
    pdf_url = None

    if isinstance(best_location, dict):
        source_url = best_location.get("landing_page_url")
        pdf_url = best_location.get("pdf_url")

    if isinstance(primary_location, dict):
        if not source_url:
            source_url = primary_location.get("landing_page_url")

        if not pdf_url:
            pdf_url = primary_location.get("pdf_url")

    # Extract Publication year
    publication_year = work.get("publication_year")

    if (
        not isinstance(publication_year, int)
        or isinstance(publication_year, bool)
    ):
        publication_year = None

    # Return normalized metadata
    return {
        "openalex_id": openalex_id,
        "title": title,
        "authors": authors,
        "abstract": abstract,
        "publication_year": publication_year,
        "doi": doi,
        "source_url": source_url,
        "pdf_url": pdf_url
    }