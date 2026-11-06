from agent.graph import build_agent_graph


def main():

    agent = build_agent_graph()

    query = input(
        "\nEnter your research topic: "
    ).strip()

    if not query:
        print("Query cannot be empty.")
        return

    initial_state = {
        "query": query,
        "route": "",
        "retrieved_chunks": [],
        "context": "",
        "research_results": [],
        "messages": [],
        "answer": "",
        "citations": [],
        "error": None,
        "document_ids": [],
        "comparison_results": []
    }

    print("\nExecuting research discovery...\n")

    result = agent.invoke(initial_state)

    if result["error"]:
        print(f"Research discovery failed: {result['error']}")
        return

    print("=" * 65)
    print("RESEARCH DISCOVERY RESULTS")
    print("=" * 65)

    print(f"\nSelected workflow: {result['route']}")
    print(f"Papers discovered: {len(result['research_results'])}")

    print("\n" + result["answer"])

    print("\n" + "=" * 65)
    print("EXECUTION COMPLETED")
    print("=" * 65)


if __name__ == "__main__":
    main()