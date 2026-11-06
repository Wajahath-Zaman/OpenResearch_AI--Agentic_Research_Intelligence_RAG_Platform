import re


class ContextGuardrail:

    def __init__(self):
        self.injection_patterns = [
            r"ignore\s+(all\s+)?previous\s+instructions",
            r"ignore\s+(all\s+)?above\s+instructions",
            r"disregard\s+(all\s+)?previous\s+instructions",
            r"reveal\s+(your\s+)?system\s+prompt",
            r"show\s+(me\s+)?your\s+system\s+prompt",
            r"override\s+(all\s+)?(previous\s+)?instructions",
            r"act\s+as\s+if\s+you\s+have\s+no\s+restrictions",
            r"bypass\s+(your\s+)?safety\s+(rules|guidelines)",
            r"follow\s+these\s+instructions\s+instead",
            r"you\s+are\s+now\s+(a|an)\s+",
        ]

    def inspect(self, retrieved_chunks: list[dict]) -> dict:

        if not isinstance(retrieved_chunks, list):
            raise TypeError("Retrieved chunks must be a list.")

        safe_chunks = []
        flagged_chunks = []

        for chunk in retrieved_chunks:

            if not isinstance(chunk, dict):
                raise TypeError("Each retrieved chunk must be a dictionary.")

            text = chunk.get("text")

            if not isinstance(text, str):
                raise ValueError("Each chunk must contain text.")

            detected_pattern = None

            for pattern in self.injection_patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    detected_pattern = pattern
                    break

            if detected_pattern:
                flagged_chunks.append({
                    "chunk": chunk,
                    "reason": "Potential prompt injection detected."
                })
            else:
                safe_chunks.append(chunk)

        return {
            "safe_chunks": safe_chunks,
            "flagged_chunks": flagged_chunks,
            "flagged_count": len(flagged_chunks),
        }