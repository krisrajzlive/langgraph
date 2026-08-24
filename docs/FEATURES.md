# LangGraph Feature Guide

This document explains the concept behind each demo in [`app/graphs/`](../app/graphs/) in more depth than the README table.

## 1. `g1_basic.py` — StateGraph fundamentals

A graph is a `TypedDict` state plus nodes (plain functions that take state and return a partial state update) wired together with edges. `START` and `END` are sentinel nodes marking entry/exit. `.compile()` turns the graph definition into a runnable.

## 2. `g2_conditional.py` — conditional edges and cycles

`add_conditional_edges(node, router_fn, path_map)` lets a node's outgoing edge depend on the current state. Pointing a conditional edge back at an earlier node creates a loop — the basis of every "keep trying until X" agent pattern.

## 3. `g3_tool_agent.py` — tool-calling agent

`MessagesState` is a built-in state shape (`{"messages": [...]}`) with an `add_messages` reducer, so new messages append instead of overwriting. `model.bind_tools([...])` lets the LLM emit `tool_calls`; `ToolNode` executes them and returns `ToolMessage`s. The `should_continue` router loops back to the model whenever a message contains tool calls, and exits otherwise — the standard ReAct loop.

## 4. `g4_memory.py` — short-term memory

A **checkpointer** (`MemorySaver` here) snapshots state after every step, keyed by `thread_id`. Calling `.invoke()` again with the same `thread_id` resumes from the last checkpoint, so the model sees the full prior conversation without the caller re-sending it.

## 5. `g5_persistence.py` — durable persistence

`SqliteSaver` is a drop-in replacement for `MemorySaver` that writes checkpoints to disk, so a thread survives process restarts. `get_state_history()` walks every checkpoint ever written for a thread — useful for time-travel debugging or auditing.

## 6. `g6_human_in_loop.py` — human-in-the-loop

Calling `interrupt(payload)` inside a node pauses the graph and surfaces `payload` to the caller via `result["__interrupt__"]`. The graph resumes exactly where it left off when invoked again with `Command(resume=<value>)` — `interrupt()` then returns that value as if it had been a normal function call. This requires a checkpointer, since resuming depends on the paused state being saved.

## 7. `g7_streaming.py` — streaming

`app.stream(inputs, stream_mode=...)` supports several modes:
- `"values"` — full state after each step
- `"updates"` — only the delta produced by each node
- `"messages"` — token-by-token chunks as the LLM generates, paired with metadata about which node produced them

## 8. `g8_subgraph.py` — subgraphs

Any compiled graph can be used directly as a node in another graph, as long as the parent and child states share the keys the subgraph reads/writes. This is how you compose reusable pipelines (e.g. a "review analysis" subgraph) into larger workflows.

## 9. `g9_parallel.py` — map-reduce fan-out with `Send`

Returning a list of `Send(node_name, state)` from a conditional-edge function dynamically spawns one parallel branch per `Send` — the number of branches doesn't need to be known until runtime. Branch outputs are merged back into the parent state via an `Annotated[list, operator.add]` reducer.

## 10. `g10_context_trimming.py` — prompt optimization: filtering + trimming

Two composable primitives from `langchain_core.messages`:
- **`filter_messages`** drops messages by type (e.g. strip tool-call noise or system messages before showing history to a summarizer).
- **`trim_messages`** caps the message list to a token/message budget, keeping the most recent turns (`strategy="last"`) and optionally re-anchoring on a human turn (`start_on="human"`) so you never start mid-tool-call.

A dedicated `optimize_context` node runs this pipeline before every model call, so the *stored* thread (`state["messages"]`) stays complete while the *prompt actually sent to the LLM* (`state["trimmed"]`) stays small — this is the same shape as a LangGraph `pre_model_hook`.

## 11. `g11_summarization.py` — context optimization: rolling summarization

Once a thread grows past a threshold, older messages are compressed into a short running summary (itself produced by an LLM call) and then physically removed from state with `RemoveMessage(id=...)` — the summary is injected as a system message on subsequent turns instead. This bounds context growth for long-running conversations without losing durable facts, at the cost of fidelity on the pruned turns.

## Concepts intentionally left out (natural next steps)

- **Prebuilt agents** (`langgraph.prebuilt.create_react_agent`) — a one-line version of demo 3.
- **Multi-agent handoff / supervisor patterns** — routing between multiple specialized agents.
- **Retry policies and node-level error handling** (`RetryPolicy`).
- **LangGraph Platform / `langgraph dev`** for a hosted graph server + Studio UI.
