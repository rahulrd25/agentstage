# Getting started

## Install

```bash
uv add agentstage
```

Or with pip:

```bash
pip install agentstage
```

Requires Python 3.12+. No Node.js, no separate frontend build — the UI ships prebuilt inside the package.

## The smallest working app

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

Run it:

```bash
python app.py
```

Open **http://127.0.0.1:8000/**. That's it — chat input, streaming output, markdown rendering, loading and error states, all without a line of frontend code.

If you already have a real LangGraph or `create_agent` agent, pass that instead of the toy graph above — everything else in this guide is unchanged.

## Adding tool-call visibility

If your agent calls tools, show that work happening:

```python
app.chat(streaming=True)
app.tool_calls(visible=True, collapsible=True)
```

Each tool call renders as a card: name, arguments, a spinner while it runs, then the result or an error. No changes needed in your agent — the adapter reads tool calls directly off LangGraph's `AIMessage.tool_calls` and correlates results by `tool_call_id`.

## Adding human-in-the-loop approval

If a node in your graph calls `interrupt()`, wire up the approval UI:

```python
app.human_approval(enabled=True)
```

This requires your agent to be compiled with a checkpointer — LangGraph cannot pause without one:

```python
from langgraph.checkpoint.memory import InMemorySaver

agent = graph.compile(checkpointer=InMemorySaver())
```

`agentstage` checks for this at startup and raises a clear error naming the fix if it's missing, rather than letting your app hang the first time a user hits the interrupt.

See [examples/human_approval/app.py](../examples/human_approval/app.py) for a full working example, including the reject path.

## Adding citations

If your model returns [LangChain's standard citation content blocks](https://docs.langchain.com/oss/python/langchain/messages), turn on rendering:

```python
app.chat(streaming=True, citations=True)
```

Citations show up as a "Sources" list under the answer. No special handling needed in your agent beyond emitting the standard `Citation` blocks LangChain already defines.

## Adding file uploads

```python
app.chat(streaming=True, attachments=True)
```

This opens a paperclip button in the composer. Uploaded files are validated (size, MIME type, and the file's real content — not just the browser-supplied `Content-Type` header) and stored server-side; only an opaque id ever appears in a chat request, never the file's bytes. Your agent receives the file as a standard LangChain image or file content block alongside the user's message.

See [examples/citations_and_files/app.py](../examples/citations_and_files/app.py).

## Adding conversation history

```python
app.thread_history(enabled=True)
```

This adds a sidebar: past conversations, switch between them, rename, delete. Like human approval, it requires a checkpointer, because that's where message content is actually read back from — `agentstage` keeps only a small metadata index (title, last-used time) separate from your checkpointer's storage.

The default index is in-memory and disappears on restart. For something durable:

```python
from agentstage.storage import SQLiteThreadStore

app = AgentApp(agent, thread_store=SQLiteThreadStore("threads.db"))
app.thread_history(enabled=True)
```

See [examples/threads_and_persistence/app.py](../examples/threads_and_persistence/app.py).

## Putting it behind your own auth

```python
def authenticate(request):
    user = my_auth_system.verify(request.headers.get("authorization"))
    if user is None:
        raise HTTPException(status_code=401)
    return user.id  # used to scope uploads and threads to this user


app = AgentApp(agent, authenticate=authenticate)
```

`authenticate` is a hook, not an implementation — it receives the raw FastAPI `Request` and may raise to reject it. Whatever it returns is used as an opaque owner key: uploads and threads are scoped so one user can never read or touch another's, but `agentstage` does not implement your authorization scheme for you.

## Mounting into an app you already have

If you already run a FastAPI service and want the agent available alongside it:

```python
from fastapi import FastAPI

host_app = FastAPI()


@host_app.get("/")
def home():
    return {"service": "my product"}


AgentApp(agent, title="Assistant").mount(host_app, prefix="/agent")
```

The agent's UI and API are now at `/agent/`, and the host app's own routes, middleware, and exception handlers are completely untouched. If you only want the JSON API — say, you're building your own frontend — use `.router()` instead of `.mount()` and skip the UI entirely:

```python
host_app.include_router(AgentApp(agent).router(), prefix="/api/agent")
```

See [examples/mount_into_existing_app/app.py](../examples/mount_into_existing_app/app.py).

## What's next

- [Architecture](architecture.md) — how the pieces fit together and why
- [Events](events.md) — the normalized event contract, if you want to build your own UI against it
- [LangGraph notes](langgraph.md) — verified API details that contradict common tutorials
- [Troubleshooting](troubleshooting.md) — common problems and fixes
- [Comparisons](comparisons.md) — how this differs from Streamlit, Reflex, and the LangChain Agent Chat UI
