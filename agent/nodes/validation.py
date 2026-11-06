from agent.state import AgentState
from guardrails.input_guardrail import InputGuardrail


def validate_input(state: AgentState, guardrail=None) -> dict:

    if guardrail is None:
        guardrail = InputGuardrail()

    result = guardrail.validate(state["query"])

    if not result["valid"]:
        return {
            "error": result["reason"],
            "route": "invalid"
        }

    return {
        "error": None
    }