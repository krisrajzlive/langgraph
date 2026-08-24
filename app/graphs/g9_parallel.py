"""Demo 9: map-reduce style fan-out/fan-in with the Send API.

Covers: dynamically spawning parallel branches at runtime with Send, an
Annotated reducer (operator.add) to merge results from parallel branches.
"""
import operator
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send


class FanState(TypedDict):
    topics: list[str]
    facts: Annotated[list[str], operator.add]


class WorkerState(TypedDict):
    topic: str
    facts: Annotated[list[str], operator.add]


FACTS = {
    "graphs": "A graph is a set of nodes connected by edges.",
    "agents": "An agent uses an LLM to decide which actions to take next.",
    "memory": "Checkpointers let a graph resume from any prior state.",
    "streaming": "Streaming emits partial results while a graph is still running.",
}


def dispatch(state: FanState) -> list[Send]:
    return [Send("lookup_fact", {"topic": topic, "facts": []}) for topic in state["topics"]]


def lookup_fact(state: WorkerState) -> dict:
    fact = FACTS.get(state["topic"], f"No fact known about '{state['topic']}'.")
    return {"facts": [f"{state['topic']}: {fact}"]}


def build_graph():
    graph = StateGraph(FanState)
    graph.add_node("lookup_fact", lookup_fact)
    graph.add_conditional_edges(START, dispatch, ["lookup_fact"])
    graph.add_edge("lookup_fact", END)
    return graph.compile()


def run(topics: list[str] | None = None) -> None:
    app = build_graph()
    topics = topics or ["graphs", "agents", "memory", "streaming"]
    result = app.invoke({"topics": topics, "facts": []})
    for fact in result["facts"]:
        print(f"- {fact}")


if __name__ == "__main__":
    run()
