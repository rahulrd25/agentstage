"""End-to-end HTTP tests: a real agent, real SSE, real FastAPI routing."""

from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from agentstage.app import AgentApp
from tests.fakes import FakeToolCallingModel, make_tool_call


@tool
def search_documents(query: str) -> str:
    """Search the document store."""
    return f"RESULT for {query}"


@tool
def boom_tool(query: str) -> str:
    """Always raises."""
    msg = f"tool exploded on {query}"
    raise ValueError(msg)


def build_agent(
    *, tools: list[Any] | None = None, answer: str = "All done.", checkpointer: Any = None
) -> Any:
    from langchain.agents import create_agent

    chosen = tools or [search_documents]
    return create_agent(
        model=FakeToolCallingModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[make_tool_call(chosen[0].name, {"query": "x"}, "call_1")],
                ),
                AIMessage(content=answer),
            ]
        ),
        tools=chosen,
        checkpointer=checkpointer,
    )


def build_agent_with_history() -> Any:
    """A checkpointed agent — thread_history() requires one to read state back."""
    from langgraph.checkpoint.memory import InMemorySaver

    return build_agent(checkpointer=InMemorySaver())


def client_for(app: AgentApp) -> TestClient:
    return TestClient(app.build())


def asset_paths(html: str) -> tuple[str, str]:
    """Extract the built JS entry point and stylesheet paths from served HTML.

    Vite fingerprints asset filenames per build (``assets/index-<hash>.js``), so
    tests cannot hardcode them and must read them out of the page itself, the
    same way a real browser would.
    """
    import re

    script = re.search(r'<script[^>]+src="([^"]+\.js)"', html)
    style = re.search(r'<link[^>]+href="([^"]+\.css)"', html)
    assert script, "no <script src> found in served HTML — did the build output change shape?"
    assert style, "no <link href> stylesheet found in served HTML — did the build change shape?"
    return script.group(1).removeprefix("./"), style.group(1).removeprefix("./")


def read_events(response: Any) -> list[dict[str, Any]]:
    """Collect the JSON payloads from an SSE response."""
    events: list[dict[str, Any]] = []
    for line in response.iter_lines():
        text = line if isinstance(line, str) else line.decode()
        if text.startswith("data:"):
            events.append(json.loads(text[5:].strip()))
    return events


def post_chat(client: TestClient, message: str, **body: Any) -> list[dict[str, Any]]:
    with client.stream("POST", "/api/chat", json={"message": message, **body}) as response:
        assert response.status_code == 200
        return read_events(response)


# ---- Health --------------------------------------------------------------


def test_health_reports_ok_and_the_title():
    client = client_for(AgentApp(build_agent(), title="Doc Assistant"))

    body = client.get("/api/health").json()

    assert body == {
        "status": "ok",
        "title": "Doc Assistant",
        "attachments": False,
        "threads": False,
    }


def test_health_reports_attachments_enabled():
    """The UI toggles its attach button from this — the alternative is always
    showing it and every click 404ing on an app that never opted in."""
    client = client_for(AgentApp(build_agent()).chat(attachments=True))

    assert client.get("/api/health").json()["attachments"] is True


def test_health_reports_threads_enabled():
    """Same reasoning as attachments: the UI toggles the sidebar from this."""
    client = client_for(AgentApp(build_agent_with_history()).thread_history(enabled=True))

    assert client.get("/api/health").json()["threads"] is True


# ---- Streaming -----------------------------------------------------------


def test_chat_returns_an_event_stream():
    client = client_for(AgentApp(build_agent()))

    with client.stream("POST", "/api/chat", json={"message": "hi"}) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")


def test_buffering_is_disabled_on_the_response():
    """Proxies buffer streams by default; this header is what prevents it."""
    client = client_for(AgentApp(build_agent()))

    with client.stream("POST", "/api/chat", json={"message": "hi"}) as response:
        assert response.headers["x-accel-buffering"] == "no"


def test_a_run_starts_and_completes_over_http():
    events = post_chat(client_for(AgentApp(build_agent())), "find x")

    kinds = [e["type"] for e in events]
    assert kinds[0] == "run_started"
    assert kinds[-1] == "run_completed"


def test_the_full_tool_lifecycle_arrives_over_the_wire():
    events = post_chat(client_for(AgentApp(build_agent())), "find x")
    kinds = [e["type"] for e in events]

    assert "tool_call_started" in kinds
    assert "tool_call_completed" in kinds

    completed = next(e for e in events if e["type"] == "tool_call_completed")
    assert completed["data"]["result"] == "RESULT for x"
    assert completed["tool_call_id"] == "call_1"


def test_the_assistant_answer_is_reassemblable_from_deltas():
    events = post_chat(client_for(AgentApp(build_agent(answer="Found it."))), "go")

    text = "".join(e["data"]["text"] for e in events if e["type"] == "message_delta")
    assert text == "Found it."


def test_sequence_numbers_arrive_gapless():
    """A client detects dropped frames by gaps, so the transport must not reorder."""
    events = post_chat(client_for(AgentApp(build_agent())), "go")

    assert [e["sequence"] for e in events] == list(range(len(events)))


def test_a_thread_id_is_generated_and_returned():
    events = post_chat(client_for(AgentApp(build_agent())), "go")

    assert events[0]["thread_id"].startswith("thread-")


def test_a_supplied_thread_id_is_used():
    events = post_chat(client_for(AgentApp(build_agent())), "go", thread_id="t-mine")

    assert events[0]["thread_id"] == "t-mine"


# ---- Failures ------------------------------------------------------------


def test_a_failing_tool_is_reported_in_band_not_as_an_http_error():
    """The response has already begun streaming, so the failure cannot be an HTTP
    status. The client learns about it as an event."""
    client = client_for(AgentApp(build_agent(tools=[boom_tool])))

    with client.stream("POST", "/api/chat", json={"message": "go"}) as response:
        assert response.status_code == 200
        events = read_events(response)

    kinds = [e["type"] for e in events]
    assert "tool_call_failed" in kinds
    assert kinds[-1] == "run_failed"


def test_a_run_terminates_exactly_once_on_failure():
    events = post_chat(client_for(AgentApp(build_agent(tools=[boom_tool]))), "go")
    kinds = [e["type"] for e in events]

    assert kinds.count("run_failed") == 1
    assert kinds.count("run_completed") == 0


def test_an_empty_message_is_rejected_before_the_agent_runs():
    client = client_for(AgentApp(build_agent()))

    assert client.post("/api/chat", json={"message": ""}).status_code == 422


def test_an_oversized_message_is_rejected():
    """An unbounded body is a denial-of-service vector."""
    client = client_for(AgentApp(build_agent()))

    response = client.post("/api/chat", json={"message": "x" * 40_000})

    assert response.status_code == 422


# ---- Security ------------------------------------------------------------


def test_no_response_carries_chain_of_thought():
    """Checked on the actual bytes sent to the browser."""
    client = client_for(AgentApp(build_agent()))

    with client.stream("POST", "/api/chat", json={"message": "go"}) as response:
        body = "".join(
            line if isinstance(line, str) else line.decode() for line in response.iter_lines()
        )

    assert "reasoning" not in body
    assert "additional_kwargs" not in body


def test_an_authenticate_hook_can_reject_a_request():
    def deny(request: Any) -> None:
        raise HTTPException(status_code=401, detail="nope")

    app = AgentApp(build_agent(), authenticate=deny)

    assert client_for(app).post("/api/chat", json={"message": "go"}).status_code == 401


def test_an_authenticate_hook_can_allow_a_request():
    calls: list[str] = []

    def allow(request: Any) -> None:
        calls.append(request.url.path)

    app = AgentApp(build_agent(), authenticate=allow)
    post_chat(client_for(app), "go")

    assert calls == ["/api/chat"]


async def test_an_async_authenticate_hook_is_awaited():
    seen: list[str] = []

    async def allow(request: Any) -> None:
        seen.append("called")

    app = AgentApp(build_agent(), authenticate=allow)
    post_chat(client_for(app), "go")

    assert seen == ["called"]


def test_health_stays_reachable_without_authentication():
    """A liveness probe that needs credentials breaks orchestrators."""

    def deny(request: Any) -> None:
        raise HTTPException(status_code=401)

    client = client_for(AgentApp(build_agent(), authenticate=deny))

    assert client.get("/api/health").status_code == 200


# ---- UI serving ----------------------------------------------------------


def test_the_ui_is_served_at_the_root():
    client = client_for(AgentApp(build_agent()))

    response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "agentstage" in response.text


def test_the_ui_assets_are_served():
    client = client_for(AgentApp(build_agent()))

    script, style = asset_paths(client.get("/").text)

    assert client.get(f"/{script}").status_code == 200
    assert client.get(f"/{style}").status_code == 200


def test_the_ui_never_shadows_the_api():
    """The static mount is at '/', so route order decides whether /api works."""
    client = client_for(AgentApp(build_agent()))

    assert client.get("/api/health").status_code == 200


def test_the_ui_is_told_where_the_api_is_mounted():
    """The UI is served from '/' but the API is under a prefix, so relative fetch
    URLs would 404. The server injects the prefix instead."""
    client = client_for(AgentApp(build_agent()))

    assert 'data-api-base="/api"' in client.get("/").text


def test_a_custom_api_prefix_reaches_the_ui():
    app = AgentApp(build_agent())
    client = TestClient(app.build(api_prefix="/agent-api"))

    assert 'data-api-base="/agent-api"' in client.get("/").text
    assert client.get("/agent-api/health").status_code == 200


def test_hidden_elements_stay_hidden_against_display_rules():
    """`hidden` is only a weak `display: none` default, so a rule like
    `.error { display: flex }` can defeat it and show an empty element on a
    fresh page. The stylesheet must force hiding to win.

    Matched with the built CSS's own token formatting (no spaces, since Vite's
    build minifies it) rather than the hand-authored source's, which would
    never match the served asset.
    """
    from agentstage.app import STATIC_DIR

    _, style = asset_paths((STATIC_DIR / "index.html").read_text(encoding="utf-8"))
    css = (STATIC_DIR / style).read_text(encoding="utf-8")
    normalized = "".join(css.split())

    assert "[hidden]{display:none!important}" in normalized


def test_the_javascript_builds_api_urls_from_the_injected_prefix():
    """Guards the specific bug this replaced: a relative './health' resolved
    against '/' and 404'd, silently breaking the title fetch.

    The API base now arrives via the ``data-api-base`` attribute the server
    rewrites into index.html (see ``AgentApp.build()``), read at runtime by
    ``frontend/src/lib/api.ts`` — so the built JS bundle is checked for that
    runtime read rather than for a baked-in path, which minification/bundling
    make unstable to match directly.
    """
    from agentstage.app import STATIC_DIR

    script, _ = asset_paths((STATIC_DIR / "index.html").read_text(encoding="utf-8"))
    source = (STATIC_DIR / script).read_text(encoding="utf-8")

    assert 'fetch("./chat"' not in source
    assert 'fetch("./health"' not in source
    assert "apiBase" in source


@pytest.mark.parametrize("bad", ["api", "", "/"])
def test_an_invalid_api_prefix_is_rejected(bad: str):
    from agentstage.errors import ConfigError

    with pytest.raises(ConfigError, match="absolute path"):
        AgentApp(build_agent()).build(api_prefix=bad)


def test_the_frontend_source_never_assigns_markup():
    """Markdown rendering is an XSS sink and tool output is untrusted, so the UI
    must build DOM nodes (JSX) rather than assign markup.

    Checked against our own authored source in frontend/src/, not the built
    bundle: React's own compiled internals legitimately use innerHTML in a few
    framework-controlled spots that have nothing to do with untrusted data, so
    grepping the minified output for these strings would fail on code we don't
    control. What actually matters is that *our* components never opt into
    dangerouslySetInnerHTML or similar.

    Comments are stripped first: a file may legitimately *mention* one of these
    sinks in prose explaining why it is avoided (e.g. markdown.tsx's docstring).
    """
    import re
    from pathlib import Path

    frontend_src = Path(__file__).resolve().parents[2] / "frontend" / "src"
    sinks = ("dangerouslySetInnerHTML", "document.write", "insertAdjacentHTML", "outerHTML =")

    sources = [
        p for p in frontend_src.rglob("*.ts*") if not p.name.endswith((".test.ts", ".test.tsx"))
    ]
    assert sources, "no frontend source files found — did the frontend move?"

    for path in sources:
        code = re.sub(r"//[^\n]*", "", path.read_text(encoding="utf-8"))
        for sink in sinks:
            assert sink not in code, f"{sink} is an XSS sink; build DOM nodes instead ({path})"
        assert "eval(" not in code, f"eval( found in {path}"


# ---- Mounting into an existing app --------------------------------------


def test_the_router_mounts_into_an_existing_fastapi_app():
    """agentstage must not require owning the user's server."""
    host = FastAPI()
    host.include_router(AgentApp(build_agent(), title="Mounted").router(), prefix="/agent")

    @host.get("/mine")
    def mine() -> dict[str, bool]:
        return {"ok": True}

    client = TestClient(host)

    assert client.get("/agent/health").json()["title"] == "Mounted"
    assert client.get("/mine").json() == {"ok": True}


def test_mount_serves_both_the_ui_and_the_api_under_the_host_apps_prefix():
    """The gap `.router()` leaves: it mounts only the JSON API, with no UI. A
    team with an existing FastAPI service that wants the chat UI available
    alongside their own routes needs the whole app — API and static assets —
    reachable under one prefix, not two separate mount calls."""
    host = FastAPI()

    @host.get("/mine")
    def mine() -> dict[str, bool]:
        return {"ok": True}

    AgentApp(build_agent(), title="Mounted").mount(host, prefix="/agent")
    client = TestClient(host)

    ui = client.get("/agent/")
    assert ui.status_code == 200
    assert "text/html" in ui.headers["content-type"]

    script, _ = asset_paths(ui.text)
    assert client.get(f"/agent/{script}").status_code == 200
    assert client.get("/agent/api/health").json()["title"] == "Mounted"
    assert client.get("/mine").json() == {"ok": True}


def test_mounted_chat_streams_through_the_full_host_app():
    """Not just that the routes exist — that a real chat turn works end to end
    through the host app's own ASGI stack, not just agentstage's standalone one."""
    host = FastAPI()
    AgentApp(build_agent()).mount(host, prefix="/agent")
    client = TestClient(host)

    with client.stream("POST", "/agent/api/chat", json={"message": "hi"}) as response:
        assert response.status_code == 200
        events = read_events(response)

    assert events[0]["type"] == "run_started"
    assert events[-1]["type"] == "run_completed"


def test_mounted_ui_is_told_the_correct_api_base_for_its_prefix():
    """The UI is served from '/agent/' but the API lives at '/agent/api' — the
    same relative-fetch bug this pattern already fixed once for the top-level
    case must not reappear one level down."""
    host = FastAPI()
    AgentApp(build_agent()).mount(host, prefix="/agent")
    client = TestClient(host)

    assert 'data-api-base="/api"' in client.get("/agent/").text


def test_a_custom_api_prefix_works_when_mounted():
    host = FastAPI()
    AgentApp(build_agent()).mount(host, prefix="/agent", api_prefix="/backend")
    client = TestClient(host)

    assert client.get("/agent/backend/health").status_code == 200
    assert 'data-api-base="/backend"' in client.get("/agent/").text


def test_the_hosts_own_routes_and_the_mounted_app_do_not_interfere():
    host = FastAPI()

    @host.get("/dashboard")
    def dashboard() -> dict[str, str]:
        return {"page": "dashboard"}

    AgentApp(build_agent()).mount(host, prefix="/agent")
    client = TestClient(host)

    assert client.get("/dashboard").json() == {"page": "dashboard"}
    assert client.get("/agent/api/health").status_code == 200


@pytest.mark.parametrize("bad_prefix", ["agent", "", "/"])
def test_an_invalid_mount_prefix_is_rejected(bad_prefix: str):
    from agentstage.errors import ConfigError

    host = FastAPI()

    with pytest.raises(ConfigError, match="absolute path"):
        AgentApp(build_agent()).mount(host, prefix=bad_prefix)


def test_mount_returns_self_for_chaining():
    host = FastAPI()
    app = AgentApp(build_agent())

    assert app.mount(host, prefix="/agent") is app


# ---- AgentApp configuration ---------------------------------------------


def test_build_is_idempotent():
    """run() and a manual uvicorn call must not build two divergent apps."""
    app = AgentApp(build_agent())

    assert app.build() is app.build()


def test_feature_methods_chain():
    app = AgentApp(build_agent()).chat(streaming=True).tool_calls(visible=True)

    assert app.build() is not None


def test_chat_citations_true_builds_successfully():
    assert AgentApp(build_agent()).chat(citations=True).build() is not None


# ---- Citations over HTTP --------------------------------------------------


def build_citing_agent() -> Any:
    """An agent whose answer carries a real Citation content block."""
    from langchain_core.messages.content import create_citation
    from langgraph.graph import END, START, MessagesState, StateGraph

    def call_model(state: MessagesState) -> dict[str, Any]:
        citation = create_citation(
            url="https://docs.example.com/api", title="API Reference", cited_text="AgentApp"
        )
        return {
            "messages": [
                AIMessage(
                    content=[
                        {
                            "type": "text",
                            "text": "See the API reference.",
                            "annotations": [citation],
                        }
                    ]
                )
            ]
        }

    graph = StateGraph(MessagesState)
    graph.add_node("model", call_model)
    graph.add_edge(START, "model")
    graph.add_edge("model", END)
    return graph.compile()


def test_a_citation_reaches_the_client_over_sse():
    client = client_for(AgentApp(build_citing_agent()).chat(citations=True))

    events = post_chat(client, "find docs")
    completed = next(e for e in events if e["type"] == "message_completed")

    assert completed["data"]["citations"] == [
        {"url": "https://docs.example.com/api", "title": "API Reference", "cited_text": "AgentApp"}
    ]


# ---- File uploads ----------------------------------------------------------


def build_echoing_agent() -> Any:
    """An agent that reports back the content blocks it received."""
    from langgraph.graph import END, START, MessagesState, StateGraph

    def call_model(state: MessagesState) -> dict[str, Any]:
        content = state["messages"][-1].content
        return {"messages": [AIMessage(content=f"received: {content!r}")]}

    graph = StateGraph(MessagesState)
    graph.add_node("model", call_model)
    graph.add_edge(START, "model")
    graph.add_edge("model", END)
    return graph.compile()


def test_upload_is_404_when_attachments_are_not_enabled():
    """Not 403 — an app that never opted in should not reveal the endpoint exists."""
    client = client_for(AgentApp(build_agent()))

    response = client.post("/api/upload", files={"file": ("a.txt", b"hi", "text/plain")})

    assert response.status_code == 404


def test_uploading_a_file_returns_an_id():
    client = client_for(AgentApp(build_agent()).chat(attachments=True))

    response = client.post("/api/upload", files={"file": ("notes.txt", b"hello", "text/plain")})

    assert response.status_code == 200
    body = response.json()
    assert body["filename"] == "notes.txt"
    assert body["size"] == 5
    assert body["id"]


def test_an_oversized_upload_is_rejected_with_422():
    from agentstage.files import InMemoryAttachmentStore

    app = AgentApp(build_agent(), attachment_store=InMemoryAttachmentStore(max_bytes=4))
    client = client_for(app.chat(attachments=True))

    response = client.post("/api/upload", files={"file": ("a.txt", b"toolong", "text/plain")})

    assert response.status_code == 422


def test_a_disallowed_type_is_rejected_with_422():
    client = client_for(AgentApp(build_agent()).chat(attachments=True))

    response = client.post(
        "/api/upload", files={"file": ("a.exe", b"MZ", "application/x-msdownload")}
    )

    assert response.status_code == 422


def test_referencing_an_uploaded_file_reaches_the_agent():
    client = client_for(AgentApp(build_echoing_agent()).chat(attachments=True))
    upload = client.post(
        "/api/upload", files={"file": ("notes.txt", b"file bytes", "text/plain")}
    ).json()

    events = post_chat(client, "look at this", attachment_ids=[upload["id"]])

    text = "".join(e["data"]["text"] for e in events if e["type"] == "message_delta")
    assert "'type': 'file'" in text
    assert "text/plain" in text


def test_referencing_an_image_upload_produces_an_image_block():
    client = client_for(AgentApp(build_echoing_agent()).chat(attachments=True))
    png_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 20
    upload = client.post(
        "/api/upload", files={"file": ("photo.png", png_bytes, "image/png")}
    ).json()

    events = post_chat(client, "what is this", attachment_ids=[upload["id"]])

    text = "".join(e["data"]["text"] for e in events if e["type"] == "message_delta")
    assert "'type': 'image'" in text


def test_referencing_an_unknown_attachment_id_is_a_404():
    client = client_for(AgentApp(build_agent()).chat(attachments=True))

    response = client.post(
        "/api/chat", json={"message": "hi", "attachment_ids": ["file-does-not-exist"]}
    )

    assert response.status_code == 404


def test_attachment_ids_are_rejected_when_attachments_are_disabled():
    client = client_for(AgentApp(build_agent()))

    response = client.post("/api/chat", json={"message": "hi", "attachment_ids": ["file-whatever"]})

    assert response.status_code == 400


def test_a_different_authenticated_owner_cannot_reference_someone_elses_upload():
    """The load-bearing security property: without this, any authenticated user
    could guess or intercept another user's attachment id and have the agent read
    their file."""

    class FakeUser:
        def __init__(self, name: str) -> None:
            self.name = name

        def __str__(self) -> str:
            return self.name

    users = iter([FakeUser("alice"), FakeUser("bob")])

    def authenticate(request: Any) -> Any:
        return next(users)

    app = AgentApp(build_agent(), authenticate=authenticate).chat(attachments=True)
    client = client_for(app)

    upload = client.post(
        "/api/upload", files={"file": ("secret.txt", b"alice's data", "text/plain")}
    ).json()

    response = client.post(
        "/api/chat", json={"message": "read this", "attachment_ids": [upload["id"]]}
    )

    assert response.status_code == 404


# ---- Thread history --------------------------------------------------------


def test_threads_is_404_when_thread_history_is_not_enabled():
    """Not an empty list — an app that never opted in should not reveal the
    endpoint exists."""
    client = client_for(AgentApp(build_agent()))

    assert client.get("/api/threads").status_code == 404


def test_a_chat_turn_creates_a_listed_thread():
    client = client_for(AgentApp(build_agent_with_history()).thread_history(enabled=True))

    post_chat(client, "hello there, this is my first message")

    threads = client.get("/api/threads").json()
    assert len(threads) == 1
    assert "hello there" in threads[0]["title"]


def test_the_thread_title_comes_from_the_first_message_only():
    client = client_for(AgentApp(build_agent_with_history()).thread_history(enabled=True))

    post_chat(client, "first message", thread_id="t1")
    post_chat(client, "a completely different second message", thread_id="t1")

    threads = client.get("/api/threads").json()
    assert len(threads) == 1
    assert "first message" in threads[0]["title"]


def test_a_long_first_message_is_truncated_for_the_title():
    client = client_for(AgentApp(build_agent_with_history()).thread_history(enabled=True))

    post_chat(client, "x" * 200, thread_id="t1")

    title = client.get("/api/threads").json()[0]["title"]
    assert len(title) <= 61  # 60 chars + ellipsis
    assert title.endswith("…")


def test_multiple_threads_are_listed_most_recently_used_first():
    client = client_for(AgentApp(build_agent_with_history()).thread_history(enabled=True))

    post_chat(client, "first thread", thread_id="t1")
    post_chat(client, "second thread", thread_id="t2")
    post_chat(client, "back to first", thread_id="t1")

    threads = client.get("/api/threads").json()
    assert [t["thread_id"] for t in threads] == ["t1", "t2"]


def test_renaming_a_thread():
    client = client_for(AgentApp(build_agent_with_history()).thread_history(enabled=True))
    post_chat(client, "hi", thread_id="t1")

    response = client.patch("/api/threads/t1", json={"title": "My renamed chat"})

    assert response.status_code == 200
    assert response.json()["title"] == "My renamed chat"
    assert client.get("/api/threads").json()[0]["title"] == "My renamed chat"


def test_renaming_an_unknown_thread_is_404():
    client = client_for(AgentApp(build_agent_with_history()).thread_history(enabled=True))

    response = client.patch("/api/threads/no-such-thread", json={"title": "x"})

    assert response.status_code == 404


def test_deleting_a_thread_removes_it_from_the_list():
    client = client_for(AgentApp(build_agent_with_history()).thread_history(enabled=True))
    post_chat(client, "hi", thread_id="t1")

    response = client.delete("/api/threads/t1")

    assert response.status_code == 204
    assert client.get("/api/threads").json() == []


def test_deleting_an_unknown_thread_is_404():
    client = client_for(AgentApp(build_agent_with_history()).thread_history(enabled=True))

    assert client.delete("/api/threads/no-such-thread").status_code == 404


def test_a_different_authenticated_owner_cannot_see_or_touch_anothers_thread():
    """The load-bearing security property, same as the attachment case: without
    owner scoping, any user could list, rename, or delete another user's chats."""

    class FakeUser:
        def __init__(self, name: str) -> None:
            self.name = name

        def __str__(self) -> str:
            return self.name

    users = iter([FakeUser("alice"), FakeUser("bob"), FakeUser("bob")])

    def authenticate(request: Any) -> Any:
        return next(users)

    app = AgentApp(build_agent_with_history(), authenticate=authenticate).thread_history(
        enabled=True
    )
    client = client_for(app)

    post_chat(client, "alice's private chat", thread_id="t1")  # as alice

    assert client.get("/api/threads").json() == []  # as bob: no threads visible
    assert client.delete("/api/threads/t1").status_code == 404  # as bob: can't delete it


# ---- Thread transcripts -----------------------------------------------------


def test_transcript_endpoint_is_404_when_thread_history_is_not_enabled():
    client = client_for(AgentApp(build_agent()))

    assert client.get("/api/threads/t1/messages").status_code == 404


def test_a_threads_transcript_reconstructs_the_conversation():
    client = client_for(AgentApp(build_agent_with_history()).thread_history(enabled=True))
    post_chat(client, "find x", thread_id="t1")

    transcript = client.get("/api/threads/t1/messages").json()

    roles = [m["role"] for m in transcript]
    assert roles[0] == "user"
    assert "tool" in roles
    assert roles[-1] == "assistant"
    assert transcript[0]["text"] == "find x"
    assert transcript[-1]["text"] == "All done."


def test_a_thread_id_unknown_to_the_thread_store_is_404():
    """The ownership check happens against the thread store; a thread_id it has
    never recorded a touch() for cannot be vouched for, so this is 404 — the same
    status as a thread owned by someone else."""
    client = client_for(AgentApp(build_agent_with_history()).thread_history(enabled=True))

    assert client.get("/api/threads/never-opened/messages").status_code == 404


async def test_a_known_thread_the_checkpointer_never_saw_returns_an_empty_transcript():
    """A gap between the two stores is possible in principle — a thread listed
    but with no checkpointer state — and must render as an empty conversation,
    not a crash."""
    from agentstage.storage import InMemoryThreadStore

    thread_store = InMemoryThreadStore()
    client = client_for(
        AgentApp(build_agent_with_history(), thread_store=thread_store).thread_history(enabled=True)
    )
    await thread_store.touch("t1", owner=None, title="listed but never run")

    assert client.get("/api/threads/t1/messages").json() == []


def test_a_different_owner_cannot_read_anothers_transcript():
    """The load-bearing security property for this endpoint: ownership is
    checked against the thread store, since the checkpointer itself has no
    concept of ownership."""

    class FakeUser:
        def __init__(self, name: str) -> None:
            self.name = name

        def __str__(self) -> str:
            return self.name

    users = iter([FakeUser("alice"), FakeUser("bob")])

    def authenticate(request: Any) -> Any:
        return next(users)

    app = AgentApp(build_agent_with_history(), authenticate=authenticate).thread_history(
        enabled=True
    )
    client = client_for(app)

    post_chat(client, "alice's private chat", thread_id="t1")  # as alice

    assert client.get("/api/threads/t1/messages").status_code == 404  # as bob


def test_thread_history_uses_a_real_sqlite_store_across_app_instances(tmp_path: Any):
    """The point of choosing SQLite over the default: a fresh AgentApp built
    after a process restart must see the same threads."""
    from agentstage.storage import SQLiteThreadStore

    db_path = tmp_path / "threads.sqlite3"
    first_app = AgentApp(
        build_agent_with_history(), thread_store=SQLiteThreadStore(db_path)
    ).thread_history(enabled=True)
    post_chat(client_for(first_app), "persisted chat", thread_id="t1")

    second_app = AgentApp(
        build_agent_with_history(), thread_store=SQLiteThreadStore(db_path)
    ).thread_history(enabled=True)
    threads = client_for(second_app).get("/api/threads").json()

    assert len(threads) == 1
    assert "persisted chat" in threads[0]["title"]


def test_thread_history_enabled_builds_successfully():
    assert AgentApp(build_agent_with_history()).thread_history(enabled=True).build() is not None


def test_human_approval_reports_the_missing_checkpointer_first():
    """The checkpointer problem is the more actionable of the two, so it wins."""
    from agentstage.errors import ConfigError

    app = AgentApp(build_agent())

    with pytest.raises(ConfigError, match="requires a checkpointer"):
        app.human_approval(enabled=True)


def test_human_approval_succeeds_with_a_checkpointer():
    from typing import TypedDict

    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.graph import END, START, StateGraph

    class State(TypedDict):
        pass

    def approve(state: State) -> State:
        from langgraph.types import interrupt

        interrupt({"question": "ok?"})
        return {}

    graph = StateGraph(State)
    graph.add_node("approve", approve)
    graph.add_edge(START, "approve")
    graph.add_edge("approve", END)
    agent = graph.compile(checkpointer=InMemorySaver())

    app = AgentApp(agent).human_approval(enabled=True)
    assert app.build() is not None


# ---- Human-in-the-loop over HTTP ------------------------------------------


def build_approval_agent() -> Any:
    """An agent that pauses for approval, then finishes once resumed."""
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.graph import END, START, MessagesState, StateGraph
    from langgraph.types import interrupt

    def approve(state: MessagesState) -> dict[str, Any]:
        decision = interrupt({"question": "Approve sending the email?"})
        return {"messages": [AIMessage(content=f"Decision recorded: {decision}")]}

    graph = StateGraph(MessagesState)
    graph.add_node("approve", approve)
    graph.add_edge(START, "approve")
    graph.add_edge("approve", END)
    return graph.compile(checkpointer=InMemorySaver())


def test_a_run_that_interrupts_streams_an_interrupt_event():
    client = client_for(AgentApp(build_approval_agent()).human_approval(enabled=True))

    events = post_chat(client, "send the email")
    kinds = [e["type"] for e in events]

    assert "interrupt_created" in kinds
    interrupt_event = next(e for e in events if e["type"] == "interrupt_created")
    assert interrupt_event["data"]["value"] == {"question": "Approve sending the email?"}
    # An interrupted run does not complete normally — it is paused, not finished.
    assert kinds[-1] != "run_completed"


def test_resume_continues_the_paused_run():
    client = client_for(AgentApp(build_approval_agent()).human_approval(enabled=True))
    first = post_chat(client, "send the email", thread_id="t-approve-1")
    thread_id = first[0]["thread_id"]

    with client.stream(
        "POST", "/api/resume", json={"thread_id": thread_id, "decision": "approved"}
    ) as response:
        assert response.status_code == 200
        events = read_events(response)

    kinds = [e["type"] for e in events]
    assert kinds[0] == "run_started"
    assert kinds[-1] == "run_completed"
    text = "".join(e["data"]["text"] for e in events if e["type"] == "message_delta")
    assert "True" in text or "true" in text.lower()


def test_reject_is_a_valid_decision_and_reaches_the_agent():
    client = client_for(AgentApp(build_approval_agent()).human_approval(enabled=True))
    first = post_chat(client, "send the email", thread_id="t-approve-2")
    thread_id = first[0]["thread_id"]

    with client.stream(
        "POST", "/api/resume", json={"thread_id": thread_id, "decision": "rejected"}
    ) as response:
        events = read_events(response)

    text = "".join(e["data"]["text"] for e in events if e["type"] == "message_delta")
    assert "False" in text or "false" in text.lower()


def test_resuming_an_unknown_thread_is_rejected_with_400():
    """Resuming a thread with no pending interrupt would silently start an
    unrelated run (verified against LangGraph) — this must 400, not 200."""
    client = client_for(AgentApp(build_approval_agent()).human_approval(enabled=True))

    response = client.post(
        "/api/resume", json={"thread_id": "no-such-thread", "decision": "approved"}
    )

    assert response.status_code == 400


def test_resuming_the_same_thread_twice_fails_the_second_time():
    client = client_for(AgentApp(build_approval_agent()).human_approval(enabled=True))
    first = post_chat(client, "send the email", thread_id="t-approve-3")
    thread_id = first[0]["thread_id"]

    with client.stream(
        "POST", "/api/resume", json={"thread_id": thread_id, "decision": "approved"}
    ):
        pass

    response = client.post("/api/resume", json={"thread_id": thread_id, "decision": "approved"})
    assert response.status_code == 400


def test_an_invalid_decision_value_is_rejected_by_validation():
    client = client_for(AgentApp(build_approval_agent()).human_approval(enabled=True))

    response = client.post("/api/resume", json={"thread_id": "whatever", "decision": "maybe"})

    assert response.status_code == 422


def test_edited_decision_carries_a_replacement_value():
    client = client_for(AgentApp(build_approval_agent()).human_approval(enabled=True))
    first = post_chat(client, "send the email", thread_id="t-approve-4")
    thread_id = first[0]["thread_id"]

    with client.stream(
        "POST",
        "/api/resume",
        json={
            "thread_id": thread_id,
            "decision": "edited",
            "edited_value": {"to": "someone-else@example.com"},
        },
    ) as response:
        events = read_events(response)

    text = "".join(e["data"]["text"] for e in events if e["type"] == "message_delta")
    assert "someone-else@example.com" in text


def test_importing_agentstage_stays_cheap():
    """A library consumer must not pay for a web stack they never start."""
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import agentstage, sys; "
            "print(','.join(m for m in ('fastapi','langgraph','uvicorn') if m in sys.modules))",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    assert result.stdout.strip() == ""


def test_agent_app_is_importable_from_the_package_root():
    from agentstage import AgentApp as Exported

    assert Exported is AgentApp


def test_run_reports_a_port_already_in_use_with_actionable_guidance():
    """A raw uvicorn OSError on a bound port is opaque; a developer running two
    servers by mistake, or wanting to add agentstage to a FastAPI app already
    listening there, should be pointed at a different port or at .mount()."""
    import socket

    from agentstage.errors import ConfigError

    blocker = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    blocker.bind(("127.0.0.1", 0))
    blocker.listen(1)
    port = blocker.getsockname()[1]
    try:
        with pytest.raises(ConfigError, match="already in use") as exc_info:
            AgentApp(build_agent()).run(host="127.0.0.1", port=port)
        assert "mount" in str(exc_info.value)
    finally:
        blocker.close()


# ---- Example app regression: shared model state across requests ----------


def test_the_basic_chat_example_shows_the_tool_call_on_every_request():
    """Regression: FakeToolCallingModel replays a fixed script and repeats the
    last entry once exhausted. In a long-lived server the *first* request spent
    the tool-call response, so every later request silently lost its tool card —
    this was invisible in tests because each test built a fresh agent, and only
    showed up when clicking through the running server by hand."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from examples.basic_chat.app import build_agent

    client = client_for(AgentApp(build_agent()))

    for _ in range(3):
        kinds = [e["type"] for e in post_chat(client, "hi")]
        assert "tool_call_started" in kinds, (
            "the tool call disappeared on a repeat request against the same app"
        )
        assert "tool_call_completed" in kinds
