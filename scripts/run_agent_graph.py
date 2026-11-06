from agent.graph import build_agent_graph
from retrieval.vector_store import ChromaVectorStore


def main():

    # Check whether documents are available
    vector_store = ChromaVectorStore()

    total_chunks = vector_store.count()

    print(f"\nAvailable document chunks: {total_chunks}")

    if total_chunks == 0:
        print("No documents available. Ingest a PDF first.")
        return

    # Build LangGraph application
    agent = build_agent_graph()

    print("\nAgent initialized successfully!")

    # Get user query
    query = input("\nEnter your research question: ").strip()

    if not query:
        print("Query cannot be empty.")
        return

    # Initial agent state
    initial_state = {
        "query": query,
        "route": "",
        "retrieved_chunks": [],
        "context": "",
        "research_results": [],
        "messages": [],
        "answer": "",
        "citations": [],
        "error": None
    }

    print("\nExecuting LangGraph workflow...\n")

    # Execute graph
    result = agent.invoke(initial_state)

    print("=" * 60)
    print("AGENT EXECUTION RESULT")
    print("=" * 60)

    print(f"\nSelected workflow: {result['route']}")

    if result["error"]:
        print(f"\nExecution error: {result['error']}")
        return

    print("\nGENERATED ANSWER")
    print("-" * 60)
    print(result["answer"])

    print("\nCITATIONS")
    print("-" * 60)

    for citation in result["citations"]:

        print(
            f"\n[{citation['citation_id']}] "
            f"{citation.get('title') or 'Untitled'}"
        )

        print(f"Page: {citation['page_number']}")
        print(f"Document ID: {citation['document_id']}")

    print("\n" + "=" * 60)
    print("LANGGRAPH EXECUTION COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()