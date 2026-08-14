# Troubleshooting

## `ConfigError: ... requires a checkpointer`

Both `app.human_approval(enabled=True)` and `app.thread_history(enabled=True)` need your agent compiled with a checkpointer — one to actually pause on `interrupt()`, the other to read message history back out. Fix:

```python
from langgraph.checkpoint.memory import InMemorySaver

agent = graph.compile(checkpointer=InMemorySaver())
```

`InMemorySaver` is fine for local development; it does not survive a process restart. See [Architecture](architecture.md#decisions-worth-understanding) for why thread *metadata* (title, last-used) is a separate, optional store from the checkpointer.

## Approval never appears — the run just hangs or completes normally

If your graph calls `interrupt()` but `human_approval` was never enabled (`app.human_approval(enabled=True)`), or the agent has no checkpointer, LangGraph cannot actually pause. This is a LangGraph-level behavior, not an agentstage bug — see [LangGraph notes](langgraph.md#checkpointers).

## Resuming an approval does nothing / starts over from the beginning

This is a real LangGraph behavior, not a bug: resuming a `thread_id` that has no pending interrupt does not raise — it silently restarts the interrupted node from its beginning. `agentstage`'s `POST /resume` checks for a pending interrupt first and returns `400` instead of letting this happen, so if you're calling the LangGraph adapter directly rather than through the HTTP API, add the same check:

```python
if await adapter.has_pending_interrupt(thread_id):
    async for event in adapter.resume(thread_id=thread_id, resume_value=True):
        ...
```

See [LangGraph notes](langgraph.md#interrupts-and-resume) for the full explanation.

## `UI assets are missing` error on startup

`AgentApp.build()` raises `ConfigError` if `src/agentstage/static/index.html` isn't found. This means the installed package is corrupted or was built incorrectly — reinstall:

```bash
pip install --force-reinstall agentstage
```

## The chat UI loads but every request 404s

Check the browser's network tab for the exact failing URL. The UI is served at `/` while the API defaults to `/api` — if you changed `api_prefix` in `.build(api_prefix=...)` or `.mount(api_prefix=...)`, make sure it's consistent between wherever your reverse proxy routes and what you passed. The UI reads its own API base from a `data-api-base` attribute the server injects into `index.html` at request time — you should never need to hardcode this yourself.

## Streaming appears to buffer — nothing shows up until the whole response finishes

This is almost always a reverse proxy (nginx, a load balancer) buffering the SSE response. `agentstage` already sends `X-Accel-Buffering: no` and `Cache-Control: no-cache, no-transform` on every streaming response, which nginx respects by default — but some proxies need buffering disabled at the proxy config level too (nginx: `proxy_buffering off;` on the relevant `location` block).

## `422 Unprocessable Entity` on `/api/chat`

`ChatRequest.message` has `min_length=1, max_length=32_000`; an empty or very long message is rejected before the agent ever runs. Check the request body against those limits.

## `422` on `/api/upload`

The `AttachmentStore` enforces a size cap, a MIME-type allowlist, and — for image and PDF types — checks the file's real magic bytes against its declared `Content-Type`, because a client can lie about that header. The response `detail` names which check failed. See [`agentstage/files.py`](../src/agentstage/files.py) for the exact limits and allowed types, which you can override by constructing your own `InMemoryAttachmentStore` with different values and passing it as `attachment_store=` to `AgentApp`.

## `404` on `/api/threads`, `/api/upload`, or `/api/resume`

These endpoints only exist if the corresponding feature is enabled — `thread_history(enabled=True)`, `chat(attachments=True)`, or a graph capable of interrupting, respectively. This is deliberate: an app that never opted into a feature shouldn't reveal that its endpoint exists at all, so a disabled feature 404s rather than 403s.

## A different user can see my threads or attachments

This would be a real security bug — please check first whether you've supplied an `authenticate` hook at all. Without one, every caller is treated as the same anonymous owner (`owner=None`), so *all* uploads and threads are visible to *any* caller — this is correct behavior for a single-user local app, not a bug, but it means multi-user deployments must supply `authenticate`. See [Getting started: putting it behind your own auth](getting-started.md#putting-it-behind-your-own-auth).

## Tests are slow or seem to need a real LLM

They shouldn't — the entire test suite runs against deterministic fakes and requires no API key. If a test you wrote is calling a real provider, use `FakeToolCallingModel` from `tests/fakes.py` instead; see [Development](../README.md#development) for why the stock LangChain fakes don't support tool calling and this one exists.
