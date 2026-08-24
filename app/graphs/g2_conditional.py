"""Demo 2: conditional edges and cycles.

Covers: add_conditional_edges, routing functions, looping a node until a
condition is met (a tiny "guess the number" agent).
"""
import random
from typing import TypedDict

from langgraph.graph import END, START, StateGraph


class GuessState(TypedDict):
    target: int
    guess: int
    attempts: int
    history: list[str]


def make_guess(state: GuessState) -> dict:
    guess = random.randint(1, 100)
    return {"guess": guess, "attempts": state["attempts"] + 1}


def record(state: GuessState) -> dict:
    diff = state["guess"] - state["target"]
    verdict = "correct" if diff == 0 else ("too high" if diff > 0 else "too low")
    entry = f"attempt {state['attempts']}: guessed {state['guess']} ({verdict})"
    return {"history": state["history"] + [entry]}


def route(state: GuessState) -> str:
    if state["guess"] == state["target"] or state["attempts"] >= 15:
        return "done"
    return "retry"


def build_graph():
    graph = StateGraph(GuessState)
    graph.add_node("make_guess", make_guess)
    graph.add_node("record", record)

    graph.add_edge(START, "make_guess")
    graph.add_edge("make_guess", "record")
    graph.add_conditional_edges("record", route, {"retry": "make_guess", "done": END})

    return graph.compile()


def run(target: int | None = None) -> None:
    app = build_graph()
    target = target if target is not None else random.randint(1, 100)
    result = app.invoke({"target": target, "guess": 0, "attempts": 0, "history": []})
    print(f"target: {target}")
    for line in result["history"]:
        print(f"  {line}")
    print(f"solved in {result['attempts']} attempts")


if __name__ == "__main__":
    run()
