# agentstage

A Python-first UI layer for LangChain and LangGraph agents.

Take an agent you already have and give it a production-quality web frontend, in Python. No React, no TypeScript, no CSS, no manual WebSocket wiring, and no Node.js on your machine.

> **Status: pre-alpha.** Chat, streaming, tool-call visibility, human approval, citations, file uploads, conversation history, and mounting into an existing FastAPI app all work end to end today — see [Project status](#project-status) for what's built per milestone. Nothing is published to PyPI yet.

**Documentation:** [Getting started](docs/getting-started.md) · [Architecture](docs/architecture.md) · [Events](docs/events.md) · [LangGraph notes](docs/langgraph.md) · [Troubleshooting](docs/troubleshooting.md) · [Comparisons](docs/comparisons.md)

## The idea

```python
from langchain.agents import create_agent
from agentstage import AgentApp

agent = create_agent(model="...", tools=[search_documents])

app = AgentApp(agent=agent, title="Document Assistant")

app.chat(streaming=True, citations=True)
app.tool_calls(visible=True, collapsible=True)
app.human_approval(enabled=True)
app.thread_history(enabled=True)
```

That should get you a working app with chat input, streaming output, markdown, visible tool calls, loading and error states, and conversation history.

## Who this is for

A Python developer who is comfortable with FastAPI, LangChain, LangGraph, LLM APIs, databases, and async Python — and who does not want to hand-write a frontend to ship an agent.

## What it is not

Not a new LLM framework, not a replacement for LangChain or LangGraph, not a BI tool, not a Streamlit clone, not a drag-and-drop builder.

## Architecture

```text
LangChain/LangGraph agent          your Python, server-side only
        │
        │  astream(stream_mode=["updates", "messages"])
        ▼
Adapter  →  normalized AgentEvent  the stable public contract
        │
        ▼
FastAPI + SSE                      text/event-stream, one-way
        │
        ▼
Browser UI                         static assets shipped in the wheel
        │
        ▼
Browser
```

Four decisions worth stating up front — full detail in [docs/architecture.md](docs/architecture.md).

**SSE, not WebSocket.** Agent streaming is one-way server→client. SSE gives auto-reconnect, plain HTTP, and debuggability with `curl`. Client→server actions — submit, approve, cancel — are ordinary `POST`s.

**The frontend ships prebuilt; the Python user never needs Node.** Built with Vite + React from `frontend/src/`, compiled to `src/agentstage/static/`, and served straight from the wheel. A Python user never runs `npm`, touches JSX, or needs Node installed — only someone changing the UI itself works in `frontend/`.

**`AgentEvent` is the insulating boundary.** LangGraph's `astream_events(version="v3")` is explicitly experimental, so the adapter is a swappable backend behind one event contract. When v3 stabilizes we add a second backend and the event model and UI code stay unchanged.

**No Reflex.** An extra Python-UI runtime was evaluated and rejected — it adds a layer without earning one, and its WebSocket-per-user model doesn't fit an SSE design.

### The normalized event model

```python
EventType = Literal[
    "run_started",
    "run_completed",
    "run_failed",
    "message_started",
    "message_delta",
    "message_completed",
    "tool_call_started",
    "tool_call_delta",
    "tool_call_completed",
    "tool_call_failed",
    "interrupt_created",
    "progress_updated",
    "state_updated",
]


@dataclass
class AgentEvent:
    type: EventType
    run_id: str
    thread_id: str | None = None
    node_name: str | None = None
    message_id: str | None = None
    tool_call_id: str | None = None
    data: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None
```

Not every LangGraph event carries every field, so all of them are optional and original metadata is preserved where it's useful.

## Verified LangGraph behavior

These were confirmed by installing the packages and introspecting them, not recalled from memory. Several contradict widely-copied tutorial code.

| Finding | Detail |
|---|---|
| `astream` versions | Accepts `version: 'v1' \| 'v2'` only. `v3` exists solely on `astream_events`. |
| `astream_events(version='v3')` | Experimental, and raises `TypeError` if you pass `stream_mode` or `subgraphs` — it owns those. |
| `StreamMode` literals | Exactly `values`, `updates`, `checkpoints`, `tasks`, `debug`, `messages`, `custom`. |
| `Interrupt` fields | Exactly two: `value` and `id`. Tutorials referencing `resumable` / `ns` / `when` will crash. |

The consequence that shapes the adapter: **token streaming and tool lifecycle arrive on different channels.** `tool_call_started` comes from the `updates` channel via `AIMessage.tool_calls` — not from `messages` — and completion is correlated through `ToolMessage.tool_call_id`. The normalizer consumes both channels and joins on `tool_call_id`.

Verified against langgraph 1.2.11, langchain 1.3.15, langchain-core 1.5.4.

## Security posture

- Agent execution stays on the server. API keys are never exposed to the browser.
- Client-submitted user IDs and tenant IDs are never trusted.
- Tool output is treated as untrusted data. Markdown rendering is sanitized — it is an XSS sink.
- Hidden chain-of-thought is never sent to the client. Only tool calls, citations, and safe execution metadata.
- No arbitrary code execution from the UI.
- Authentication is a **hook interface**, not an implementation. Row-level authorization is the application's responsibility and will be documented as such. The MVP does not pretend to have complete authorization.

## Development

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync                        # install, including dev dependencies
uv run pytest                  # run the Python test suite
uv run ruff check . && uv run ruff format --check .
uv run mypy                    # strict on src/
```

No API key is needed to develop or test. The suite runs against a deterministic fake model and fake tools; no test requires OpenAI or any paid API.

One piece of test infrastructure is worth knowing about: **no stock LangChain fake model supports `bind_tools`** (`GenericFakeChatModel` raises `NotImplementedError`), so tool-lifecycle tests are impossible without one. agentstage ships a small `FakeToolCallingModel` in its test fakes to close that gap.

The frontend (`frontend/`) has its own test suite, covering component behavior directly in TypeScript rather than through the built Python package:

```bash
cd frontend
npm install
npm run test                   # Vitest — component and rendering regression tests
npm run build                  # tsc -b && vite build — also the package's static assets
npm run lint                   # oxlint
```

## Project status

Built:

- **Milestone 0 — architecture.** Decision recorded above, plus the LangGraph API ground truth, verified by introspection.
- **Milestone 1 — package skeleton.** Package configuration and verified dependency pins; `agentstage.types` (event/stream literals and the `SupportsAgentStream` protocol); `agentstage.errors` (exception hierarchy); ruff + mypy (strict on `src/`) + pytest configuration; `FakeToolCallingModel`.
- **Milestone 2 — event model.** `agentstage.events.AgentEvent`: frozen, validated, JSON round-tripping, with typed constructors per event type, `describe()` for logs, and `sanitized` to strip reasoning at the transport boundary. Backward compatibility is enforced by a golden-file contract test covering all 13 event types — verified to actually fail on a field rename.
- **Milestone 3 — LangGraph adapter.** `StreamNormalizer` joins the `updates` and `messages` channels into ordered, sequence-stamped events, correlating tool calls by `tool_call_id`. `LangGraphAdapter` drives a real compiled agent, guarantees every run ends in exactly one `run_completed` or `run_failed`, attributes propagating tool exceptions to the open call, and detects a missing checkpointer before a run rather than hanging.

- **Milestone 4 — transport and UI.** `AgentApp` (the public API), a mountable FastAPI router with SSE streaming and an authentication hook, and a UI serving chat with streaming markdown, collapsible tool cards, loading/error/empty states, stop, retry, and clear.
- **Milestone 6 — human-in-the-loop.** `LangGraphAdapter.resume()` and `has_pending_interrupt()`; `POST /api/resume`; an approval card in the UI with working Approve/Reject buttons. Guards the verified LangGraph footgun where resuming a thread with no pending interrupt silently starts an unrelated run instead of erroring.
- **Milestone 7 — citations and files.** Citations extracted from LangChain's real `Citation` content blocks and rendered as a sources list. File uploads via `POST /api/upload`, referenced by id from `POST /api/chat` — never round-tripped as base64 in the chat body — with a size cap, a MIME allowlist, and magic-byte verification against a lying `Content-Type` header. Owner-scoped so one user cannot reference another's upload.
- **Milestone 8 — threads and persistence.** `agentstage.storage.ThreadStore`: a metadata index (title, last-used, owner) kept separate from the checkpointer, which stays the source of truth for message content. `InMemoryThreadStore` and a `SQLiteThreadStore` (stdlib `sqlite3` via `asyncio.to_thread` — no new dependency) satisfy the same contract, checked against both in one parametrized test suite. `GET /threads`, rename, delete, and `GET /threads/{id}/messages` to reconstruct a past conversation. A sidebar in the UI lists, switches between, renames, and deletes conversations.
- **Milestone 9 — FastAPI integration.** `AgentApp.mount(host_app, prefix="/agent")` puts the whole app — API and chat UI — under a prefix of an existing FastAPI service, alongside its own routes and its own auth, with no interference in either direction (verified with a real chat turn streaming through the host's ASGI stack). `AgentApp.router()` remains for teams that want only the JSON API with their own frontend.
- **Milestone 10 — documentation.** [docs/getting-started.md](docs/getting-started.md), [docs/architecture.md](docs/architecture.md), [docs/events.md](docs/events.md), [docs/langgraph.md](docs/langgraph.md), [docs/troubleshooting.md](docs/troubleshooting.md), and [docs/comparisons.md](docs/comparisons.md). Every code sample in the getting-started guide is verified to actually run, not hand-typed and hoped correct; every claim in the LangGraph notes was re-checked against the installed packages while writing this milestone, not carried forward from memory.

All ten original milestones are complete. 314 Python tests plus a Vitest suite for the frontend, no API key required.

### Seeing it work

```bash
uv run python examples/basic_chat/app.py                # streaming chat + tool calls
uv run python examples/human_approval/app.py             # approve/reject before a risky action
uv run python examples/citations_and_files/app.py        # sources list + file upload
uv run python examples/threads_and_persistence/app.py    # conversation sidebar, SQLite-backed
uv run python examples/mount_into_existing_app/app.py    # mounted into a host app's own auth
```

Then open **http://127.0.0.1:8000/** for whichever one you ran. Each is a complete app in well under 100 lines of Python, with no frontend code.

To watch the raw event stream in a terminal instead:

```bash
uv run python examples/stream_to_terminal.py
```

### What's not built yet

No automated browser (Playwright-style) end-to-end test suite, no rate limiting, and no built-in observability/tracing integration.

Work proceeds in small, independently testable milestones. Public API stability is a commitment only once the API is actually introduced.

## License

[MIT](LICENSE)
