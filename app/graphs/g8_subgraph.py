"""Demo 8: subgraphs - a compiled graph used as a node in a parent graph.

Covers: composing graphs, shared/overlapping state keys between parent and
child, and invoking a subgraph transparently as part of a larger flow.
"""
from typing import TypedDict

from langgraph.graph import END, START, StateGraph


class ReviewState(TypedDict):
    text: str
    word_count: int
    sentiment: str


class PipelineState(TypedDict):
    text: str
    word_count: int
    sentiment: str
    summary: str


def count_words(state: ReviewState) -> dict:
    return {"word_count": len(state["text"].split())}


def naive_sentiment(state: ReviewState) -> dict:
    positive_words = {"good", "great", "love", "excellent", "happy", "amazing"}
    negative_words = {"bad", "terrible", "hate", "poor", "sad", "awful"}
    tokens = {w.strip(".,!?").lower() for w in state["text"].split()}
    score = len(tokens & positive_words) - len(tokens & negative_words)
    sentiment = "positive" if score > 0 else ("negative" if score < 0 else "neutral")
    return {"sentiment": sentiment}


def build_review_subgraph():
    sub = StateGraph(ReviewState)
    sub.add_node("count_words", count_words)
    sub.add_node("naive_sentiment", naive_sentiment)
    sub.add_edge(START, "count_words")
    sub.add_edge("count_words", "naive_sentiment")
    sub.add_edge("naive_sentiment", END)
    return sub.compile()


def summarize(state: PipelineState) -> dict:
    summary = f"{state['word_count']} words, sentiment: {state['sentiment']}"
    return {"summary": summary}


def build_graph():
    review_subgraph = build_review_subgraph()

    graph = StateGraph(PipelineState)
    graph.add_node("review", review_subgraph)  # compiled graph used directly as a node
    graph.add_node("summarize", summarize)

    graph.add_edge(START, "review")
    graph.add_edge("review", "summarize")
    graph.add_edge("summarize", END)

    return graph.compile()


def run(text: str = "This product is great, I love how it works and the support team was amazing.") -> None:
    app = build_graph()
    result = app.invoke({"text": text, "word_count": 0, "sentiment": "", "summary": ""})
    print(f"text:    {text}")
    print(f"summary: {result['summary']}")


if __name__ == "__main__":
    run()
