# Architecture

```text
LangChain/LangGraph agent          your Python, server-side only
        │
        │  astream(stream_mode=["updates", "messages"])
        ▼
LangGraphAdapter  →  normalized AgentEvent      the stable public contract
        │
        ▼
FastAPI + SSE                      text/event-stream, one-way
        │
        ▼
Browser UI                         static assets shipped in the wheel
```

## Layers

**`agentstage.adapters.LangGraphAdapter`** drives your compiled agent and turns its raw stream into normalized events. It owns run lifecycle: every run ends in exactly one `run_completed` or `run_failed`, a propagating tool exception is attributed to the open tool call before the run is marked failed, and a missing checkpointer is detected before a run starts rather than causing a silent hang.

**`agentstage.events.StreamNormalizer`** does the actual channel-joining. LangGraph splits one logical conversation across two channels that must be consumed together — see [Events](events.md) for why.

**`agentstage.runtime.fastapi`** is the HTTP surface: `/chat`, `/resume`, `/upload`, `/threads`. Pydantic models validate every request. An `authenticate` hook — not an implementation — lets an application map a request to a user without agentstage inventing its own auth system.

**`agentstage.app.AgentApp`** is the public API. Its feature methods (`chat()`, `tool_calls()`, `human_approval()`, `thread_history()`) are configuration, not construction — they record intent and return `self`, so call order never matters and nothing is built until `.build()`, `.run()`, or `.mount()` actually assembles the FastAPI app.

**The browser UI** lives in `src/agentstage/static/` as plain HTML/CSS/JS with no build step and no framework runtime — shipped straight from the wheel. This was a deliberate choice for getting the full pipeline (agent → events → browser) working and proven before investing in a richer frontend toolchain; the event contract and HTTP surface are what make that swap possible later without touching Python user code.

## Decisions worth understanding

**SSE, not WebSocket.** Agent streaming is one-way, server→client. SSE gives auto-reconnect, plain HTTP, and `curl`-level debuggability for free. Client→server actions — submit a message, approve an interrupt, cancel a run — are ordinary `POST`s; there is no bidirectional channel to manage.

**`AgentEvent` is the insulating boundary.** LangGraph's `astream_events(version="v3")` is explicitly experimental as of the versions this was built against. The adapter is a swappable backend behind one stable event contract — when a new streaming API stabilizes, a second adapter can be added without changing the event model or any UI code.

**Thread content and thread metadata are different stores.** LangGraph's checkpointer is already the durable, authoritative store for message content per thread — resuming an interrupt and reconstructing a past conversation both read from it directly. What it has no concept of is a title, a last-used timestamp, or an owner for listing purposes. `agentstage.storage.ThreadStore` is exactly that thin index, kept deliberately separate so it never duplicates what the checkpointer already owns. See [`agentstage/storage.py`](../src/agentstage/storage.py).

**Attachments never round-trip as bytes through the chat body.** A file is uploaded once (`POST /upload`), validated (size, MIME allowlist, and its real content checked against the declared `Content-Type` — a client can lie about that header), and stored server-side under an opaque id. A chat request carries only that id; the adapter resolves it to bytes just before building the LangGraph input. See [`agentstage/files.py`](../src/agentstage/files.py).

## What agentstage is not

Not a new LLM framework, not a replacement for LangChain or LangGraph, not a general BI tool, not a Streamlit clone, not a visual drag-and-drop builder, not a complete authorization system. See [Comparisons](comparisons.md) for how it relates to adjacent tools.

## Security posture

- Agent execution stays on the server. API keys are never exposed to the browser.
- Client-submitted identity (a `thread_id`, an `attachment_id`) is never trusted as an authorization token by itself — see the `authenticate` hook.
- Tool output and uploaded file content are treated as untrusted data. The markdown renderer builds real DOM nodes and never assigns `innerHTML`; a tool returning `<img onerror=...>` renders as literal text.
- Hidden chain-of-thought is never sent to the client — `AgentEvent.sanitized` strips known reasoning keys at the transport boundary, as defense in depth even if a caller forgets.
- Authentication is a hook interface, not an implementation. Row-level authorization (which threads or uploads a given user may touch) is enforced by agentstage's own storage layers using the key your `authenticate` hook returns, but the hook itself — verifying who the caller actually is — is your application's responsibility.
