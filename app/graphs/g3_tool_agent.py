"""Demo 3: a tool-calling ReAct-style agent, built by hand (not the prebuilt helper).

Covers: MessagesState, binding tools to a chat model, ToolNode, conditional
routing based on tool_calls, and a model<->tools loop.
"""
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from app.config import get_chat_model
from app.tools.basic_tools import ALL_TOOLS


def build_graph():
    model = get_chat_model().bind_tools(ALL_TOOLS)
    tool_node = ToolNode(ALL_TOOLS)

    def call_model(state: MessagesState) -> dict:
        response = model.invoke(state["messages"])
        return {"messages": [response]}

    def should_continue(state: MessagesState) -> str:
        last = state["messages"][-1]
        return "tools" if getattr(last, "tool_calls", None) else "end"

    graph = StateGraph(MessagesState)
    graph.add_node("agent", call_model)
    graph.add_node("tools", tool_node)

    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
    graph.add_edge("tools", "agent")

    return graph.compile()


def run(question: str = "What's the weather in Bengaluru and what is 12 * (4 + 1)?") -> None:
    app = build_graph()
    result = app.invoke({"messages": [{"role": "user", "content": question}]})
    for msg in result["messages"]:
        role = msg.__class__.__name__
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        print(f"[{role}] {content}")


if __name__ == "__main__":
    run()
