"""Demo 5: durable persistence via SqliteSaver.

Covers: persisting checkpoints to disk so a conversation survives across
separate process runs, plus inspecting state history with get_state_history.
"""
import os

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, MessagesState, StateGraph

from app.config import CHECKPOINT_DB_PATH, get_chat_model


def build_graph(saver: SqliteSaver):
    model = get_chat_model()

    def call_model(state: MessagesState) -> dict:
        response = model.invoke(state["messages"])
        return {"messages": [response]}

    graph = StateGraph(MessagesState)
    graph.add_node("agent", call_model)
    graph.add_edge(START, "agent")
    graph.add_edge("agent", END)

    return graph.compile(checkpointer=saver)


def run() -> None:
    os.makedirs(os.path.dirname(CHECKPOINT_DB_PATH) or ".", exist_ok=True)
    thread = {"configurable": {"thread_id": "durable-thread-1"}}

    with SqliteSaver.from_conn_string(CHECKPOINT_DB_PATH) as saver:
        app = build_graph(saver)

        result = app.invoke(
            {"messages": [{"role": "user", "content": "Remember the code word 'PineappleRocket'."}]},
            config=thread,
        )
        print(f"agent> {result['messages'][-1].content}\n")

        history = list(app.get_state_history(thread))
        print(f"{len(history)} checkpoint(s) stored in {CHECKPOINT_DB_PATH}")

        result = app.invoke(
            {"messages": [{"role": "user", "content": "What was the code word?"}]},
            config=thread,
        )
        print(f"agent> {result['messages'][-1].content}")


if __name__ == "__main__":
    run()
