"""CLI entrypoint - run any LangGraph feature demo by number or name.

Usage:
    uv run main.py                # list all demos
    uv run main.py 1              # run demo 1
    uv run main.py basic          # run by name
    uv run main.py all            # run every demo in sequence
"""
import sys

from rich.console import Console
from rich.panel import Panel

console = Console()

DEMOS = [
    ("basic", "StateGraph 101: nodes, edges, START/END", "app.graphs.g1_basic"),
    ("conditional", "Conditional edges + looping until a condition holds", "app.graphs.g2_conditional"),
    ("tool_agent", "ReAct-style tool-calling agent (needs Ollama)", "app.graphs.g3_tool_agent"),
    ("memory", "Short-term memory via MemorySaver + thread_id (needs Ollama)", "app.graphs.g4_memory"),
    ("persistence", "Durable persistence via SqliteSaver (needs Ollama)", "app.graphs.g5_persistence"),
    ("human_in_loop", "Human-in-the-loop approval via interrupt()", "app.graphs.g6_human_in_loop"),
    ("streaming", "Streaming: updates + token-level (needs Ollama)", "app.graphs.g7_streaming"),
    ("subgraph", "Subgraphs: a compiled graph nested as a node", "app.graphs.g8_subgraph"),
    ("parallel", "Fan-out/fan-in with the Send API (map-reduce)", "app.graphs.g9_parallel"),
    ("trimming", "Context optimization: filter + trim messages (needs Ollama)", "app.graphs.g10_context_trimming"),
    ("summarization", "Context optimization: rolling summary + pruning (needs Ollama)", "app.graphs.g11_summarization"),
]


def list_demos() -> None:
    console.print(Panel.fit("LangGraph Feature Demos", style="bold cyan"))
    for i, (name, desc, _) in enumerate(DEMOS, start=1):
        console.print(f"  [bold]{i}[/bold]. [green]{name}[/green] - {desc}")
    console.print("\nRun with: [bold]uv run main.py <number|name|all>[/bold]")


def run_demo(name: str, module_path: str) -> None:
    console.print(Panel.fit(f"Running demo: {name}", style="bold magenta"))
    module = __import__(module_path, fromlist=["run"])
    module.run()
    console.print()


def resolve(arg: str):
    if arg.isdigit():
        idx = int(arg) - 1
        if 0 <= idx < len(DEMOS):
            return DEMOS[idx]
        return None
    for demo in DEMOS:
        if demo[0] == arg:
            return demo
    return None


def main() -> None:
    if len(sys.argv) < 2:
        list_demos()
        return

    arg = sys.argv[1]

    if arg == "all":
        for name, _, module_path in DEMOS:
            run_demo(name, module_path)
        return

    demo = resolve(arg)
    if demo is None:
        console.print(f"[red]Unknown demo:[/red] {arg}")
        list_demos()
        sys.exit(1)

    run_demo(demo[0], demo[2])


if __name__ == "__main__":
    main()
