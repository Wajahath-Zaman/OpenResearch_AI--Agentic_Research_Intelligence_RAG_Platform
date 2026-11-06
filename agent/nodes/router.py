from agent.state import AgentState
from agent.dependencies import create_intent_classifier


def route_query(state: AgentState, intent_classifier=None) -> dict:

    query = state["query"].strip()

    if not query:
        return {
            "route": "",
            "error": "Query cannot be empty."
        }

    # Try LLM-based classification first.
    try:
        if intent_classifier is None:
            intent_classifier = create_intent_classifier()

        intent = intent_classifier.classify(query)

        return {
            "route": intent,
            "error": None
        }

    except Exception:
        # Fall back to keyword-based routing.
        query_lower = query.lower()

        discovery_keywords = (
            "find papers",
            "find research",
            "search papers",
            "discover papers",
            "recent papers",
            "latest papers"
        )

        if any(keyword in query_lower for keyword in discovery_keywords):
            route = "research"
        else:
            route = "rag"

        return {
            "route": route,
            "error": None
        }