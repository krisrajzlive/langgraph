"""Demo 4: short-term memory via checkpointing (MemorySaver + thread_id).

Covers: in-memory checkpointer, threads/conversations, multi-turn state that
persists across separate .invoke() calls within the same thread_id.
"""
from langgraph.checkpoint.memory import MemorySaver
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

    return graph.compile(checkpointer=MemorySaver())


def run() -> None:
    app = build_graph()
    thread = {"configurable": {"thread_id": "demo-conversation-1"}}

    turns = [
        "My name is Kris and my favorite language is Python.",
        "What's my name?",
        "What language do I like?",
    ]
    for turn in turns:
        result = app.invoke({"messages": [{"role": "user", "content": turn}]}, config=thread)
        print(f"user > {turn}")
        print(f"agent> {result['messages'][-1].content}\n")


if __name__ == "__main__":
    run()
