"""Demo 7: streaming graph execution.

Covers: stream_mode="values" (full state after each step), stream_mode="updates"
(only the node's delta), and token-level streaming with stream_mode="messages".
"""
from langgraph.graph import END, START, MessagesState, StateGraph

from app.config import get_chat_model


def build_graph():
    model = get_chat_model()

    def call_model(state: MessagesState) -> dict:
        response = model.invoke(state["messages"])
        return {"messages": [response]}

    graph = StateGraph(MessagesState)
    graph.add_node("agent", call_model)
    graph.add_edge(START, "agent")
    graph.add_edge("agent", END)

    return graph.compile()


def run(question: str = "Name three uses of graphs in software, briefly.") -> None:
    app = build_graph()
    inputs = {"messages": [{"role": "user", "content": question}]}

    print("--- stream_mode='updates' (per-node deltas) ---")
    for update in app.stream(inputs, stream_mode="updates"):
        for node, payload in update.items():
            print(f"[{node}] -> {payload['messages'][-1].content[:80]}...")

    print("\n--- stream_mode='messages' (token-level) ---")
    for chunk, _metadata in app.stream(inputs, stream_mode="messages"):
        text = getattr(chunk, "content", "")
        if text:
            print(text, end="", flush=True)
    print()


if __name__ == "__main__":
    run()
