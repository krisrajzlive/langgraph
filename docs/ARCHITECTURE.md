# Architecture

```
                     +-------------------+
                     |  main.py (CLI)    |
                     |  dispatches demo  |
                     +---------+---------+
                               |
                               v
        +----------------------------------------------+
        |            app/graphs/g*.py                   |
        |  each module builds + compiles one StateGraph |
        +----------------------------------------------+
                 |                          |
                 v                          v
     +---------------------+     +--------------------------+
     |  app/config.py      |     |  app/tools/basic_tools.py |
     |  get_chat_model()   |     |  offline demo tools       |
     |  -> ChatOllama       |     +--------------------------+
     +----------+-----------+
                |
                v
     +---------------------+
     |  Ollama server        |
     |  (local or container) |
     +---------------------+
```

## Runtime modes

**Local**: `main.py` runs in the host Python/uv environment and talks to an Ollama process already running on `localhost:11434` (`OLLAMA_BASE_URL` in `.env`).

**Containerized** (`docker-compose.yml`): two services —
- `ollama` — the official Ollama image, model weights persisted in a named volume (`ollama_data`)
- `app` — this project, built from the `Dockerfile`, pointed at `http://ollama:11434` via `OLLAMA_BASE_URL`

The app container depends on the `ollama` service's healthcheck (`ollama list` succeeding) before it starts.

## Persistence

Checkpointers are the only stateful piece:
- `MemorySaver` (demo 4, 6, 10, 11) — in-process, lost when the process exits. Fine for a single CLI run.
- `SqliteSaver` (demo 5) — writes to `data/checkpoints.sqlite`, mounted as a volume in Docker so state survives container restarts.

## Why Ollama instead of a hosted LLM API

The brief called for a fully local stack with no external API keys. `langchain-ollama`'s `ChatOllama` is a drop-in `BaseChatModel`, so every LangGraph pattern here (tool calling, streaming, structured state) works identically against a hosted model — swapping `get_chat_model()` in `app/config.py` for `ChatOpenAI`/`ChatAnthropic`/etc. requires no other code changes.
