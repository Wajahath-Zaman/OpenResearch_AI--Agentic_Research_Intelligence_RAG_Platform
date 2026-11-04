from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievalConfig:
    default_k: int = 5
    max_k: int = 20

    def __post_init__(self):

        if self.default_k <= 0:
            raise ValueError("Default K must be greater than zero.")

        if self.max_k <= 0:
            raise ValueError("Maximum K must be greater than zero.")

        if self.default_k > self.max_k:
            raise ValueError("Default K cannot exceed maximum K.")

    def resolve_k(self, requested_k: int | None = None) -> int:
        """
        Resolve and validate the requested number of retrieval results.
        """

        if requested_k is None:
            return self.default_k

        if not isinstance(requested_k, int) or isinstance(requested_k, bool):
            raise TypeError("Requested K must be an integer.")

        if requested_k <= 0:
            raise ValueError("Requested K must be greater than zero.")

        if requested_k > self.max_k:
            raise ValueError(
                f"Requested K cannot exceed maximum K ({self.max_k})."
            )

        return requested_k