# LangGraph Demo

A hands-on tour of [LangGraph](https://langchain-ai.github.io/langgraph/) features, built with:

- **Python 3.12** + **[uv](https://docs.astral.sh/uv/)** for dependency management
- **[LangGraph](https://github.com/langchain-ai/langgraph)** + **[LangChain](https://github.com/langchain-ai/langchain)** for the graph/agent runtime
- **[Ollama](https://ollama.com/)** as a fully local LLM backend (no API keys required)
- **Docker / docker-compose** to run the whole stack (app + Ollama) in containers

Every LangGraph concept is isolated in its own file under [`app/graphs/`](app/graphs/) so you can read, run, and modify each one independently.

## Features covered

| # | Demo | LangGraph concepts |
|---|------|---------------------|
| 1 | `basic` | `StateGraph`, `TypedDict` state, nodes, edges, `START`/`END` |
| 2 | `conditional` | `add_conditional_edges`, routing functions, cycles/loops |
| 3 | `tool_agent` | `MessagesState`, tool binding, `ToolNode`, agent<->tools loop |
| 4 | `memory` | Short-term memory: `MemorySaver`, threads (`thread_id`) |
| 5 | `persistence` | Durable memory: `SqliteSaver`, `get_state_history` |
| 6 | `human_in_loop` | `interrupt()`, `Command(resume=...)`, human approval gates |
| 7 | `streaming` | `stream_mode="updates"` and `"messages"` (token streaming) |
| 8 | `subgraph` | Composing a compiled graph as a node in a parent graph |
| 9 | `parallel` | `Send` API for dynamic map-reduce fan-out/fan-in |
| 10 | `trimming` | **Prompt optimization**: `filter_messages`, `trim_messages`, a pre-model context-shrinking node |
| 11 | `summarization` | **Context optimization**: rolling summarization + `RemoveMessage` pruning |

Run `uv run main.py` with no arguments to see this list live.

## Prerequisites

- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- [Ollama](https://ollama.com/download), running locally, with a model pulled:
  ```bash
  ollama pull llama3.2:3b
  ```
- Docker + Docker Compose (only needed for the containerized path)

Demos 1, 2, 6, 8, and 9 are pure graph-logic demos and don't call the LLM at all. Demos 3, 4, 5, 7, 10, and 11 call the local Ollama model.

## Running locally

```bash
uv sync
cp .env.example .env   # adjust OLLAMA_BASE_URL / OLLAMA_MODEL if needed

uv run main.py               # list all demos
uv run main.py basic         # run one demo by name
uv run main.py 3             # ...or by number
uv run main.py all           # run every demo in sequence
```

## Running with Docker

This starts an `ollama` container plus the app container in the same compose network:

```bash
docker compose up --build
```

The first run will need a model pulled into the `ollama` container:

```bash
docker compose exec ollama ollama pull llama3.2:3b
```

By default the container runs `main.py all`. Override the command to run a specific demo:

```bash
docker compose run --rm app trimming
```

## Project layout

```
app/
  config.py              # Ollama connection + model factory
  tools/basic_tools.py   # offline demo tools (calculator, mock weather, word count)
  graphs/
    g1_basic.py .. g11_summarization.py   # one file per feature demo
main.py                  # CLI dispatcher for the demos above
docs/                    # additional notes (architecture, feature guide)
data/                    # sqlite checkpoint DB written by the persistence demo (gitignored)
docker-compose.yml       # app + ollama services
Dockerfile
```

See [`docs/FEATURES.md`](docs/FEATURES.md) for a deeper explanation of each concept and [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for how the pieces fit together.
