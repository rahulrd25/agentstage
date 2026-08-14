# agentstage

![Python](https://img.shields.io/badge/python-3.12%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-pre--alpha-orange)

A Python-first UI layer for LangChain and LangGraph agents.

agentstage takes an agent you have already built and gives it a production-quality web frontend, written entirely in Python. There is no React, no TypeScript, no CSS, and no manual WebSocket wiring to write, and no Node.js required on your machine to run it.

```python
from langchain.agents import create_agent
from agentstage import AgentApp

agent = create_agent(model="...", tools=[search_documents])

app = AgentApp(agent=agent, title="Document Assistant")

app.chat(streaming=True, citations=True)
app.tool_calls(visible=True, collapsible=True)
app.human_approval(enabled=True)
app.thread_history(enabled=True)

app.run()
```

That gives you a working app with chat input, streaming output, markdown rendering, visible tool calls, loading and error states, and conversation history.

## Features

- **Streaming chat.** Token-level streaming over Server-Sent Events, with markdown rendering on the client.
- **Tool call visibility.** Tool invocations and their results are shown in the UI as they happen, collapsible and configurable.
- **Human-in-the-loop approval.** Pause a run at a LangGraph interrupt and resume it from an approve or reject action in the UI.
- **Citations.** Sources attached to a message through LangChain's `Citation` content blocks are rendered as a sources list.
- **File uploads.** Upload files for an agent to use, with a size cap, a MIME allowlist, and content verification independent of the declared content type.
- **Conversation history.** Threads are listed, renamed, and deleted from a sidebar, backed by an in-memory or SQLite store.
- **Mountable.** Mount the whole app, or just its JSON API, into an existing FastAPI service under a prefix, alongside your own routes and authentication.

## Installation

agentstage is not yet published to PyPI. In the meantime, install it directly from GitHub:

```bash
pip install git+https://github.com/rahulrd25/agentstage.git
```

Once published, installation will be:

```bash
pip install agentstage
```

Requires Python 3.12 or later.

## Quick start

```python
from langgraph.graph import END, START, MessagesState, StateGraph
from langchain_core.messages import AIMessage

from agentstage import AgentApp


def call_model(state: MessagesState) -> dict:
    return {"messages": [AIMessage(content="Hello from your agent!")]}


graph = StateGraph(MessagesState)
graph.add_node("model", call_model)
graph.add_edge(START, "model")
graph.add_edge("model", END)
agent = graph.compile()

app = AgentApp(agent, title="My First Agent")
app.chat(streaming=True)
app.run()
```

Then open `http://127.0.0.1:8000/`. See [docs/getting-started.md](docs/getting-started.md) for the full walkthrough.

## Documentation

- [Getting started](docs/getting-started.md)
- [Architecture](docs/architecture.md)
- [Events](docs/events.md)
- [LangGraph notes](docs/langgraph.md)
- [Troubleshooting](docs/troubleshooting.md)
- [Comparisons with Streamlit and Reflex](docs/comparisons.md)

## Examples

Each example is a complete, runnable app in well under 100 lines of Python, with no frontend code to write.

```bash
uv run python examples/basic_chat/app.py                # streaming chat and tool calls
uv run python examples/human_approval/app.py             # approve or reject before a risky action
uv run python examples/citations_and_files/app.py        # sources list and file upload
uv run python examples/threads_and_persistence/app.py    # conversation sidebar, SQLite-backed
uv run python examples/mount_into_existing_app/app.py    # mounted into a host app's own auth
```

## How it works

```text
LangChain/LangGraph agent          your Python, server-side only
        |
        |  astream(stream_mode=["updates", "messages"])
        v
Adapter  ->  normalized AgentEvent  the stable public contract
        |
        v
FastAPI + SSE                      text/event-stream, one-way
        |
        v
Browser UI                         static assets shipped in the wheel
```

Agent execution and all API keys stay on the server. The adapter normalizes LangGraph's event stream into a stable `AgentEvent` contract, decoupling the UI from LangGraph's own streaming API, which is still experimental. Client-to-server actions such as submitting a message, approving an interrupt, or canceling a run are ordinary HTTP requests; only the agent's output streams back over SSE.

See [docs/architecture.md](docs/architecture.md) for the full set of design decisions and the reasoning behind them.

## Who this is for

A Python developer comfortable with FastAPI, LangChain, LangGraph, LLM APIs, and async Python, who does not want to hand-write a frontend to ship an agent.

agentstage is not a general-purpose app framework, not a replacement for LangChain or LangGraph, not a BI tool, and not a drag-and-drop builder. See [docs/comparisons.md](docs/comparisons.md) for how it relates to Streamlit and Reflex.

## Security

- Agent execution runs on the server only. API keys are never exposed to the browser.
- Client-submitted user IDs and tenant IDs are never trusted.
- Tool output is treated as untrusted data; markdown rendering is sanitized against it.
- Hidden chain-of-thought is never sent to the client, only tool calls, citations, and safe execution metadata.
- There is no arbitrary code execution from the UI.
- Authentication is exposed as a hook interface, not an implementation. Row-level authorization remains the responsibility of the host application.

## Project status

agentstage is pre-alpha. Chat, streaming, tool call visibility, human approval, citations, file uploads, conversation history, and mounting into an existing FastAPI app all work end to end today, and are covered by an automated test suite. The public API is not yet guaranteed stable between releases.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, running the test suite, and the project layout.

## License

[MIT](LICENSE)
