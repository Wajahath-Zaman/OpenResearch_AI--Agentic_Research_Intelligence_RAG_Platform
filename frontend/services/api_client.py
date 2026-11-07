import requests

from frontend.config import API_BASE_URL, REQUEST_TIMEOUT


class APIError(Exception):
    pass


class APIClient:

    def __init__(self):
        self.base_url = API_BASE_URL.rstrip("/")
        self.session = requests.Session()

    def _request(self, method, endpoint, **kwargs):
        url = f"{self.base_url}{endpoint}"

        try:
            response = self.session.request(
                method=method,
                url=url,
                timeout=REQUEST_TIMEOUT,
                **kwargs
            )

            response.raise_for_status()

            if not response.content:
                return None

            try:
                return response.json()
            except ValueError as error:
                raise APIError(
                    "The backend returned an invalid JSON response."
                ) from error

        except requests.exceptions.HTTPError as error:
            response = error.response

            if response is not None:
                try:
                    detail = response.json().get(
                        "detail", response.text
                    )
                except ValueError:
                    detail = response.text
            else:
                detail = str(error)

            raise APIError(f"Backend error: {detail}") from error

        except requests.exceptions.ConnectionError as error:
            raise APIError(
                "Cannot connect to the backend. "
                "Make sure FastAPI is running."
            ) from error

        except requests.exceptions.Timeout as error:
            raise APIError(
                "The backend request timed out. Please try again."
            ) from error

        except requests.exceptions.RequestException as error:
            raise APIError(
                f"Communication error: {error}"
            ) from error

    def health_check(self):
        return self._request("GET", "/health")

    def list_documents(self, skip=0, limit=100):
        return self._request(
            "GET", "/documents/",
            params={"skip": skip, "limit": limit}
        )

    def get_document(self, document_id):
        return self._request("GET", f"/documents/{document_id}")

    def delete_document(self, document_id):
        return self._request("DELETE", f"/documents/{document_id}")

    def upload_document(self, uploaded_file):
        return self._request(
            "POST",
            "/documents/upload",
            files={
                "file": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    "application/pdf"
                )
            }
        )

    def ingest_document(self, document_id):
        return self._request(
            "POST", f"/documents/{document_id}/ingest"
        )

    def discover_research(
        self, query, per_page=10, page=1, save_to_database=True
    ):
        return self._request(
            "POST",
            "/research/discover",
            json={
                "query": query,
                "per_page": per_page,
                "page": page,
                "save_to_database": save_to_database
            }
        )

    def chat(self, query, document_ids=None):
        return self._request(
            "POST",
            "/chat/",
            json={
                "query": query,
                "document_ids": document_ids or []
            }
        )

    def compare_papers(self, query, document_ids):
        return self._request(
            "POST",
            "/research/compare",
            json={
                "query": query,
                "document_ids": document_ids
            }
        )

    def download_document(self, document_id):
        return self._request(
            "POST",
            f"/documents/{document_id}/download"
        )