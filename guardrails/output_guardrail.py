import re


class OutputGuardrail:

    def __init__(self, max_length: int = 10000):
        self.max_length = max_length

        self.sensitive_patterns = [
            r"(?i)api[_-]?key\s*[:=]\s*\S+",
            r"(?i)secret[_-]?key\s*[:=]\s*\S+",
            r"(?i)password\s*[:=]\s*\S+",
            r"(?i)bearer\s+[a-zA-Z0-9._-]{20,}",
        ]

    def validate(self, answer: str) -> dict:

        if not isinstance(answer, str):
            return {
                "valid": False,
                "reason": "Response must be a string."
            }

        answer = answer.strip()

        if not answer:
            return {
                "valid": False,
                "reason": "Response cannot be empty."
            }

        if len(answer) > self.max_length:
            return {
                "valid": False,
                "reason": "Response exceeds the maximum allowed length."
            }

        for pattern in self.sensitive_patterns:
            if re.search(pattern, answer):
                return {
                    "valid": False,
                    "reason": "Potential sensitive information exposure detected."
                }

        return {
            "valid": True,
            "reason": None
        }