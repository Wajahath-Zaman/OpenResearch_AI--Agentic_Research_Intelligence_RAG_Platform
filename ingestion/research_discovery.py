import requests

from config.settings import get_settings


class OpenAlexAPIError(Exception):
    """Raised when OpenAlex API request Fails."""


class OpenAlexClient:

    def __init__(self, timeout: int = 20):

        settings = get_settings()

        self.base_url = settings.openalex_base_url.rstrip("/")
        self.api_key = settings.openalex_api_key
        self.timeout = timeout

        self.session = requests.Session()
    
    def search_works(self, query:str, per_page = 10, page: int = 1) -> dict:

        # Validate Search Query
        if not query or not query.strip():
            raise ValueError("Search query Cannot be empty.")

        # Validate Pagination
        if not 1 <= per_page <= 100:
            raise ValueError("per_page must be between 1 and 100.")

        if page < 1:
            raise ValueError("Page must be greater than or equal to 1.")

        # Construc Endpoint
        endpoint = f"{self.base_url}/works"

        # Construct request parameters
        params = {
            "search": query.strip(),
            "per_page": per_page,
            "page": page
        }

        if self.api_key:
            params["api_key"] = self.api_key


        try:
            response = self.session.get(
                endpoint,
                params=params,
                timeout=self.timeout
            )

            response.raise_for_status()

            data = response.json()

            if not isinstance(data, dict):
                raise OpenAlexAPIError(
                    "Invalid Response Format From OpenAlex."
                )

            if not isinstance(data.get("results"), list):
                raise OpenAlexAPIError(
                    "OpenAlex Response does not contain valid results."
                )

            return data

        except requests.RequestException as exc:

            raise OpenAlexAPIError(
                f"OpenAlex request failed: {exc}"
            ) from exc

        except ValueError as exc:

            raise OpenAlexAPIError(
                "OpenAlex Returned invalid JSON."
            ) from exc
