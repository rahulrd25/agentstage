"""agentstage mounted into a pre-existing FastAPI application.

    uv run python examples/mount_into_existing_app/app.py

The "host" application below is meant to represent a real service your team
already runs — it has its own routes, its own auth, and its own docs, none of
which agentstage touches. The agent is available at /agent/ alongside all of it.

Routes that work in a browser, no auth needed:

    http://127.0.0.1:8000/               — the host app's own route
    http://127.0.0.1:8000/docs           — the host app's own OpenAPI docs
    http://127.0.0.1:8000/agent/api/docs — agentstage's own OpenAPI docs

This example's auth is header-based (an X-API-Key header) to demonstrate
reusing a host's existing service-to-service auth — and a plain browser tab
cannot attach a custom header to its own page load, so /agent/ itself will 401
if you just click it. That is a property of this example's chosen auth
mechanism, not a limitation of mounting. Drive it with curl instead:

    curl -H "X-API-Key: demo" -N -X POST http://127.0.0.1:8000/agent/api/chat \\
        -H "Content-Type: application/json" -d '{"message": "hello"}'

A cookie-based auth hook (session cookies, set by a login page the host app
already serves) works in a real browser tab without this caveat, because the
browser attaches cookies automatically.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Header, HTTPException
from langchain_core.messages import AIMessage
from langgraph.graph import END, START, MessagesState, StateGraph

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agentstage import AgentApp


def build_agent() -> Any:
    def call_model(state: MessagesState) -> dict[str, Any]:
        last = state["messages"][-1]
        return {"messages": [AIMessage(content=f"Echo: {last.content}")]}

    graph = StateGraph(MessagesState)
    graph.add_node("model", call_model)
    graph.add_edge(START, "model")
    graph.add_edge("model", END)
    return graph.compile()


# ---- The host application ---------------------------------------------------
# Stand-ins for whatever your real service already has: its own routes, and its
# own way of identifying a caller. agentstage's authenticate hook plugs directly
# into a FastAPI Request, so it can call the same dependency the host already
# uses instead of a parallel auth system.


def require_api_key(x_api_key: Annotated[str | None, Header()] = None) -> str:
    """Stand-in for the host's real auth. Any non-empty key is "valid" here —
    swap this for whatever your service already verifies against."""
    if not x_api_key:
        raise HTTPException(status_code=401, detail="X-API-Key header is required.")
    return str(x_api_key)


host_app = FastAPI(title="Example Product API")


@host_app.get("/")
def home() -> dict[str, str]:
    return {"service": "Example Product API", "agent_ui": "/agent/"}


@host_app.get("/dashboard")
def dashboard(api_key: Annotated[str, Depends(require_api_key)]) -> dict[str, str]:
    return {"page": "dashboard", "caller": api_key}


def authenticate_for_agent(request: Any) -> str:
    """agentstage's hook reuses the host's own auth check, rather than
    reimplementing key-checking a second time for the agent's routes.

    Header-based auth is realistic for a service-to-service API, but the
    agentstage UI is a browser page and cannot attach a custom header to its own
    requests — that's a genuine limitation, not one this example papers over.
    See the module docstring for how to actually try the UI here.
    """
    api_key: str | None = request.headers.get("x-api-key")
    if not api_key:
        raise HTTPException(status_code=401, detail="X-API-Key header is required.")
    return api_key


AgentApp(build_agent(), title="Product Assistant", authenticate=authenticate_for_agent).mount(
    host_app, prefix="/agent"
)


def main() -> None:
    import uvicorn

    print("Host app:     http://127.0.0.1:8000/")
    print("Agent docs:   http://127.0.0.1:8000/agent/api/docs")
    print()
    print("The agent's own UI needs an X-API-Key header a browser tab can't send;")
    print("see this file's module docstring for a curl command that works.")
    uvicorn.run(host_app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
