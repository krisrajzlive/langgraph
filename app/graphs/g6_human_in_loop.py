"""Demo 6: human-in-the-loop approval via interrupt().

Covers: the interrupt() primitive, pausing a run mid-graph, inspecting the
interrupt payload, and resuming with Command(resume=...).
"""
from typing import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt


class PurchaseState(TypedDict):
    item: str
    price: float
    approved: bool
    result: str


def request_approval(state: PurchaseState) -> dict:
    decision = interrupt(
        {
            "question": f"Approve purchase of '{state['item']}' for ${state['price']:.2f}?",
            "item": state["item"],
            "price": state["price"],
        }
    )
    return {"approved": bool(decision)}


def finalize(state: PurchaseState) -> dict:
    if state["approved"]:
        return {"result": f"Purchased '{state['item']}' for ${state['price']:.2f}."}
    return {"result": f"Purchase of '{state['item']}' was rejected."}


def build_graph():
    graph = StateGraph(PurchaseState)
    graph.add_node("request_approval", request_approval)
    graph.add_node("finalize", finalize)

    graph.add_edge(START, "request_approval")
    graph.add_edge("request_approval", "finalize")
    graph.add_edge("finalize", END)

    return graph.compile(checkpointer=MemorySaver())


def run() -> None:
    app = build_graph()
    thread = {"configurable": {"thread_id": "purchase-1"}}

    result = app.invoke({"item": "GPU", "price": 899.0, "approved": False, "result": ""}, config=thread)

    if "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        print(f"[interrupt] {payload['question']}")
        print("(auto-approving for demo purposes)")
        result = app.invoke(Command(resume=True), config=thread)

    print(f"result: {result['result']}")


if __name__ == "__main__":
    run()
