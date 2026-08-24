"""Demo 11: context optimization via rolling summarization.

Covers: compressing old conversation turns into a running summary once the
thread grows past a threshold, using RemoveMessage to prune the raw history
out of state while keeping its gist - the classic LangGraph pattern for
keeping long-running conversations within the model's context window.
"""
from langchain_core.messages import RemoveMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph

from app.config import get_chat_model

SUMMARY_TRIGGER = 6  # once the thread exceeds this many messages, summarize the oldest ones
KEEP_RECENT = 2  # always keep this many most-recent messages verbatim


class SummaryState(MessagesState):
    summary: str


def call_model(state: SummaryState) -> dict:
    model = get_chat_model()
    context = state["messages"]
    if state.get("summary"):
        context = [{"role": "system", "content": f"Conversation summary so far: {state['summary']}"}] + context
    response = model.invoke(context)
    return {"messages": [response]}


def maybe_summarize(state: SummaryState) -> dict:
    if len(state["messages"]) <= SUMMARY_TRIGGER:
        return {}

    model = get_chat_model()
    to_compress = state["messages"][:-KEEP_RECENT]
    prior_summary = state.get("summary", "")

    prompt = (
        f"Existing summary: {prior_summary or '(none)'}\n\n"
        "Extend it with the new messages below in 1-2 sentences. "
        "Be terse, keep only durable facts.\n\n"
        + "\n".join(f"{m.type}: {m.content}" for m in to_compress)
    )
    new_summary = model.invoke(prompt).content

    remove_ops = [RemoveMessage(id=m.id) for m in to_compress]
    return {"summary": new_summary, "messages": remove_ops}


def build_graph():
    graph = StateGraph(SummaryState)
    graph.add_node("agent", call_model)
    graph.add_node("maybe_summarize", maybe_summarize)

    graph.add_edge(START, "agent")
    graph.add_edge("agent", "maybe_summarize")
    graph.add_edge("maybe_summarize", END)

    return graph.compile(checkpointer=MemorySaver())


def run() -> None:
    app = build_graph()
    thread = {"configurable": {"thread_id": "summary-demo-1"}}

    turns = [
        "Hi, I'm planning a trip to Japan.",
        "I want to visit Tokyo and Kyoto.",
        "My budget is around $3000.",
        "I'll travel in the first week of November.",
        "Can you remind me what my budget was and where I'm going?",
    ]
    for turn in turns:
        result = app.invoke({"messages": [{"role": "user", "content": turn}]}, config=thread)

    state = app.get_state(thread)
    print(f"final raw message count in state: {len(state.values['messages'])}")
    print(f"rolling summary: {state.values.get('summary', '(none yet)')}")
    print(f"agent> {result['messages'][-1].content}")


if __name__ == "__main__":
    run()
