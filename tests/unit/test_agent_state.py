from agent.state import AgentState


def test_agent_state_structure():

    state: AgentState = {
        "query": "Explain Vision Transformers",
        "route": "rag",
        "retrieved_chunks": [],
        "context": "",
        "research_results": [],
        "messages": [],
        "answer": "",
        "citations": [],
        "error": None
    }

    assert state["query"] == "Explain Vision Transformers"
    assert state["route"] == "rag"
    assert state["retrieved_chunks"] == []
    assert state["context"] == ""
    assert state["research_results"] == []
    assert state["messages"] == []
    assert state["answer"] == ""
    assert state["citations"] == []
    assert state["error"] is None


def test_agent_state_supports_error():

    state: AgentState = {
        "query": "Find research papers",
        "route": "research",
        "retrieved_chunks": [],
        "context": "",
        "research_results": [],
        "messages": [],
        "answer": "",
        "citations": [],
        "error": "Research API unavailable"
    }

    assert state["error"] == "Research API unavailable"