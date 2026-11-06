from langgraph.graph import StateGraph, START, END

from agent.state import AgentState
from agent.nodes.router import route_query
from agent.nodes.workflows import (
    rag_workflow,
    research_workflow,
    comparison_workflow
)

from agent.nodes.validation import validate_input
from guardrails.input_guardrail import InputGuardrail

def build_agent_graph(
        answer_generator=None, 
        discovery_service=None,
        intent_classifier=None,
        comparison_handler=None,
        input_guardrail=None
    ):

    graph = StateGraph(AgentState)

    if intent_classifier is not None:
        graph.add_node(
            "router",
            lambda state: route_query(
                state,
                intent_classifier=intent_classifier
            )
        )
    else:
        graph.add_node("router", route_query)

    if answer_generator is not None:
        graph.add_node(
            "rag",
            lambda state: rag_workflow(
                state,
                answer_generator=answer_generator
            )
        )
    else:
        graph.add_node("rag", rag_workflow)

    if discovery_service is not None:
        graph.add_node(
            "research",
            lambda state: research_workflow(
                state,
                discovery_service=discovery_service
            )
        )
    else:
        graph.add_node("research", research_workflow)

    graph.add_edge(START, "validation")

    graph.add_conditional_edges(
        "validation",
        lambda state: "router" if state["error"] is None else "end",
        {
            "router": "router",
            "end": END
        }
    )

    graph.add_conditional_edges(
        "router",
        lambda state: state["route"],
        {
            "rag": "rag",
            "research": "research",
            "comparison": "comparison"
        }
    )

    if comparison_handler is not None:
        graph.add_node(
            "comparison",
            lambda state: comparison_workflow(
                state,
                comparison_handler=comparison_handler
            )
        )
    else:
        graph.add_node("comparison", comparison_workflow)


    if input_guardrail is not None:
        graph.add_node(
            "validation",
            lambda state: validate_input(
                state,
                guardrail=input_guardrail
            )
        )
    else:
        graph.add_node("validation", validate_input)
        

    graph.add_edge("rag", END)
    graph.add_edge("research", END)
    graph.add_edge("comparison", END)

    return graph.compile()