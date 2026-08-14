# Contributing

Thanks for your interest in improving agentstage. This document covers how to set up the project, run the test suite, and understand how the pieces fit together for development purposes.

## Requirements

- Python 3.12 or later
- [uv](https://docs.astral.sh/uv/)
- Node.js 20 or later (only needed if you are changing the frontend)

## Setup

```bash
uv sync
```

This installs the package along with its development dependencies (pytest, ruff, mypy, uvicorn).

No API key is required to develop or test agentstage. The test suite runs against a deterministic fake model and fake tools, so nothing requires a live connection to an LLM provider.

## Running checks

```bash
uv run pytest                              # test suite
uv run ruff check . && uv run ruff format --check .   # lint and formatting
uv run mypy                                # type checking, strict on src/
```

All three must pass before a pull request is merged.

### Test fakes

No stock LangChain fake model supports `bind_tools` (`GenericFakeChatModel` raises `NotImplementedError`), so tool-lifecycle tests are otherwise impossible to write. `tests/fakes.py` ships a small `FakeToolCallingModel` that closes this gap and is used throughout the test suite.

## Working on the frontend

The browser UI is built with Vite and React from `frontend/src/`, then compiled into `src/agentstage/static/` and served directly from the installed package. Python users never run `npm`, touch JSX, or need Node installed; only someone changing the UI itself works inside `frontend/`.

```bash
cd frontend
npm install
npm run test     # Vitest: component and rendering tests
npm run build    # tsc -b && vite build; also produces the package's static assets
npm run lint     # oxlint
```

`src/agentstage/static/` is git-ignored and rebuilt from `frontend/` as part of the release process. If you change anything under `frontend/src/`, run `npm run build` before testing the change through the Python package.

## Project layout

```
src/agentstage/
  app.py            AgentApp, the public entry point
  adapters/         LangGraphAdapter and the streaming integration
  events/           AgentEvent model and the stream normalizer
  runtime/          FastAPI router and SSE transport
  storage.py        Thread persistence (in-memory and SQLite backends)
  files.py          File upload handling
  static/           Compiled frontend, served from the wheel

frontend/           React/TypeScript source for the UI
docs/               Architecture, events, and usage documentation
examples/           Runnable end-to-end example apps
tests/              Unit and integration test suites
```

See [docs/architecture.md](docs/architecture.md) for how these pieces communicate, and [docs/events.md](docs/events.md) for the event contract that connects the adapter, transport, and UI layers.

## LangGraph API notes

This project tracks LangGraph's streaming and interrupt APIs closely, and several details in those APIs are undocumented or contradict widely copied tutorial code. Before relying on LangGraph behavior that isn't already covered in [docs/langgraph.md](docs/langgraph.md), verify it against the installed package rather than assuming it from memory or existing tutorials, and update that document if you find something new.

## Development history

Work has proceeded in small, independently testable milestones:

- **Milestone 0, architecture.** Core design decisions and LangGraph API behavior, verified by introspection rather than assumed from tutorials.
- **Milestone 1, package skeleton.** Package configuration, dependency pins, `agentstage.types`, `agentstage.errors`, and the lint/type/test toolchain.
- **Milestone 2, event model.** `agentstage.events.AgentEvent`: frozen, validated, JSON round-tripping, with typed constructors per event type and a golden-file contract test covering all 13 event types.
- **Milestone 3, LangGraph adapter.** `StreamNormalizer` joins the `updates` and `messages` channels into ordered events, correlating tool calls by `tool_call_id`. `LangGraphAdapter` guarantees every run ends in exactly one `run_completed` or `run_failed`.
- **Milestone 4, transport and UI.** `AgentApp`, a mountable FastAPI router with SSE streaming, and a UI with streaming markdown, collapsible tool cards, and loading/error/empty states.
- **Milestone 6, human-in-the-loop.** `LangGraphAdapter.resume()`, `has_pending_interrupt()`, and an approval flow in the UI.
- **Milestone 7, citations and files.** Citation extraction and file uploads with a size cap, MIME allowlist, and magic-byte verification.
- **Milestone 8, threads and persistence.** `agentstage.storage.ThreadStore` with in-memory and SQLite backends, plus a UI sidebar for managing conversations.
- **Milestone 9, FastAPI integration.** `AgentApp.mount()` for embedding into an existing FastAPI service without interfering with its routes or auth.
- **Milestone 10, documentation.** The guides under `docs/`, each checked against the installed packages rather than carried forward from memory.

All ten original milestones are complete. The test suite currently covers 314 Python tests plus a separate Vitest suite for the frontend.

Not yet built: an automated browser end-to-end test suite, rate limiting, and a built-in observability or tracing integration.

## Submitting changes

1. Open an issue describing the change first for anything larger than a small fix, so the approach can be discussed before you invest time in it.
2. Keep pull requests focused on a single change.
3. Add or update tests for any behavior change.
4. Update relevant documentation in `docs/` if the change affects the public API or event contract.
