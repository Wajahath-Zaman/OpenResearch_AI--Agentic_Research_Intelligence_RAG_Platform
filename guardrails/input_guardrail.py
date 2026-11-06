import re

class InputGuardrail:

    def __init__(self, max_length: int = 2000):
        self.max_length = max_length

    def validate(self, query: str) -> dict:

        if not isinstance(query, str):
            return {
                "valid": False,
                "reason": "Query must be a string."
            }

        query = query.strip()

        if not query:
            return {
                "valid": False,
                "reason": "Query cannot be empty."
            }

        if len(query) > self.max_length:
            return {
                "valid": False,
                "reason": "Query exceeds the maximum allowed length."
            }

        return {
            "valid": True,
            "reason": None
        }


class InputGuardrail:

    def __init__(self, max_length: int = 2000):
        self.max_length = max_length

        self.injection_patterns = [
            r"ignore\s+(all\s+)?previous\s+instructions",
            r"ignore\s+(all\s+)?above\s+instructions",
            r"disregard\s+(all\s+)?previous\s+instructions",
            r"reveal\s+(your\s+)?system\s+prompt",
            r"show\s+(me\s+)?your\s+system\s+prompt",
            r"act\s+as\s+if\s+you\s+have\s+no\s+restrictions",
            r"bypass\s+(your\s+)?safety\s+(rules|guidelines)",
            r"override\s+(all\s+)?(previous\s+)?instructions"
        ]

    def validate(self, query: str) -> dict:

        if not isinstance(query, str):
            return {
                "valid": False,
                "reason": "Query must be a string."
            }

        query = query.strip()

        if not query:
            return {
                "valid": False,
                "reason": "Query cannot be empty."
            }

        if len(query) > self.max_length:
            return {
                "valid": False,
                "reason": "Query exceeds the maximum allowed length."
            }

        for pattern in self.injection_patterns:

            if re.search(pattern, query, re.IGNORECASE):
                return {
                    "valid": False,
                    "reason": "Potential prompt injection detected."
                }

        return {
            "valid": True,
            "reason": None
        }