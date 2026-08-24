"""Demo 10: prompt/context optimization - trimming and filtering messages.

Covers: trim_messages() to cap history by token count before it hits the
model, filter_messages() to drop noisy message types (e.g. tool chatter),
and a "pre_model_hook"-style node that runs this optimization on every turn
so the prompt sent to the LLM stays small even as the thread grows.
"""
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, filter_messages, trim_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph

from app.config import get_chat_model

SYSTEM_PROMPT = SystemMessage("You are a concise assistant. Answer in one short sentence.")
MAX_TOKENS = 4  # small on purpose, so the demo visibly drops older turns


def optimize_context(state: MessagesState) -> dict:
    """Filter out tool noise, then trim to a token budget - runs before every model call."""
    filtered = filter_messages(state["messages"], exclude_types=["tool", "system"])
    trimmed = trim_messages(
        filtered,
        token_counter=len,  # cheap approximation: 1 "token" per message for this demo
        max_tokens=MAX_TOKENS,
        strategy="last",
        start_on="human",
        include_system=False,
    )
    return {"trimmed": [SYSTEM_PROMPT, *trimmed]}


class OptimizedState(MessagesState):
    trimmed: list


def build_graph():
    model = get_chat_model()

    def call_model(state: OptimizedState) -> dict:
        response = model.invoke(state["trimmed"])
        return {"messages": [response]}

    graph = StateGraph(OptimizedState)
    graph.add_node("optimize_context", optimize_context)
    graph.add_node("agent", call_model)

    graph.add_edge(START, "optimize_context")
    graph.add_edge("optimize_context", "agent")
    graph.add_edge("agent", END)

    return graph.compile(checkpointer=MemorySaver())


def run() -> None:
    app = build_graph()
    thread = {"configurable": {"thread_id": "trim-demo-1"}}

    filler_turns = [
        "Fact 1: the sky is blue.",
        "Fact 2: water boils at 100C at sea level.",
        "Fact 3: Python was created by Guido van Rossum.",
        "Fact 4: LangGraph models workflows as graphs.",
        "Fact 5: Ollama runs LLMs locally.",
    ]
    for turn in filler_turns:
        app.invoke({"messages": [HumanMessage(turn)], "trimmed": []}, config=thread)

    result = app.invoke(
        {"messages": [HumanMessage("How many facts did I just tell you? Give your best guess.")], "trimmed": []},
        config=thread,
    )

    state = app.get_state(thread)
    full_history_len = len(state.values["messages"])
    trimmed_len = len(state.values["trimmed"])

    print(f"full thread history: {full_history_len} messages")
    print(f"trimmed context sent to model: {trimmed_len} messages (system + last N)")
    last = result["messages"][-1]
    content = last.content if isinstance(last, AIMessage) else str(last)
    print(f"agent> {content}")


if __name__ == "__main__":
    run()
