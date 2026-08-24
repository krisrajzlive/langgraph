"""Demo 1: the simplest possible LangGraph - a linear StateGraph.

Covers: StateGraph, TypedDict state, add_node, add_edge, START/END, invoke().
"""
from typing import TypedDict

from langgraph.graph import END, START, StateGraph


class GreetingState(TypedDict):
    name: str
    greeting: str
    shout: str


def make_greeting(state: GreetingState) -> dict:
    return {"greeting": f"Hello, {state['name']}!"}


def shout_it(state: GreetingState) -> dict:
    return {"shout": state["greeting"].upper() + "!!!"}


def build_graph():
    graph = StateGraph(GreetingState)
    graph.add_node("make_greeting", make_greeting)
    graph.add_node("shout_it", shout_it)

    graph.add_edge(START, "make_greeting")
    graph.add_edge("make_greeting", "shout_it")
    graph.add_edge("shout_it", END)

    return graph.compile()


def run(name: str = "LangGraph") -> None:
    app = build_graph()
    result = app.invoke({"name": name, "greeting": "", "shout": ""})
    print(f"greeting -> {result['greeting']}")
    print(f"shout    -> {result['shout']}")


if __name__ == "__main__":
    run()
