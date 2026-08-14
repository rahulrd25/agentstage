"""The public API: :class:`AgentApp`.

    from agentstage import AgentApp

    app = AgentApp(agent=agent, title="Research Assistant")
    app.chat(streaming=True)
    app.tool_calls()
    app.run()

Feature methods are configuration, not construction: they record intent and
return ``self``, so ordering never matters and nothing is built until the app is
served. That keeps the FastAPI app a single object built once, in one place.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

from agentstage.adapters.langgraph import LangGraphAdapter
from agentstage.errors import ConfigError
from agentstage.files import AttachmentStore, InMemoryAttachmentStore
from agentstage.runtime.fastapi import AppConfig, build_router
from agentstage.storage import InMemoryThreadStore, ThreadStore
from agentstage.types import SupportsAgentStream

if TYPE_CHECKING:  # pragma: no cover - import cost avoided at runtime
    from fastapi import FastAPI

__all__ = ["AgentApp"]

#: Compiled UI assets, shipped in the wheel. The Python user never needs Node.
STATIC_DIR = Path(__file__).parent / "static"


@dataclass
class ChatOptions:
    """Chat feature flags.

    ``attachments`` is accepted now and lands with file upload; enabling it
    raises rather than silently doing nothing, so no one ships believing it's on.
    """

    enabled: bool = True
    streaming: bool = True
    citations: bool = False
    attachments: bool = False


@dataclass
class ToolCallOptions:
    enabled: bool = True
    visible: bool = True
    collapsible: bool = True


@dataclass
class ApprovalOptions:
    enabled: bool = False


@dataclass
class ThreadHistoryOptions:
    enabled: bool = False


class AgentApp:
    """A web application for a LangChain/LangGraph agent.

    The agent is validated immediately: a graph that was never compiled fails here
    rather than when a user first sends a message.
    """

    def __init__(
        self,
        agent: SupportsAgentStream,
        *,
        title: str = "Agent",
        authenticate: Any = None,
        run_config: dict[str, Any] | None = None,
        attachment_store: AttachmentStore | None = None,
        thread_store: ThreadStore | None = None,
    ) -> None:
        self.adapter = LangGraphAdapter(agent)
        self.title = title
        self.authenticate = authenticate
        self.run_config = run_config or {}
        #: Built lazily so a process that never enables the corresponding
        #: feature never allocates a store it will not use.
        self._attachment_store = attachment_store
        self._thread_store = thread_store
        self._chat = ChatOptions()
        self._tool_calls = ToolCallOptions()
        self._approval = ApprovalOptions()
        self._threads = ThreadHistoryOptions()
        self._app: FastAPI | None = None

    # ---- Feature configuration -------------------------------------------

    def chat(
        self,
        *,
        streaming: bool = True,
        citations: bool = False,
        attachments: bool = False,
    ) -> Self:
        """Configure the chat surface.

        ``attachments=True`` opens ``POST /upload`` and lets ``POST /chat``
        reference uploaded files. The default store is in-memory and per-process
        — pass ``attachment_store`` to the constructor for anything durable or
        multi-worker.
        """
        self._chat = ChatOptions(
            enabled=True, streaming=streaming, citations=citations, attachments=attachments
        )
        return self

    def tool_calls(self, *, visible: bool = True, collapsible: bool = True) -> Self:
        """Configure tool-call display."""
        self._tool_calls = ToolCallOptions(enabled=True, visible=visible, collapsible=collapsible)
        return self

    def human_approval(self, *, enabled: bool = True) -> Self:
        """Enable human-in-the-loop approval.

        Verifies the agent has a checkpointer, because LangGraph cannot interrupt
        without one and the symptom is a run that silently never pauses.
        """
        if enabled:
            self.adapter.require_checkpointer()
        self._approval = ApprovalOptions(enabled=enabled)
        return self

    def thread_history(self, *, enabled: bool = True) -> Self:
        """Enable a thread list: ``GET /threads``, rename, delete, and reading a
        past conversation's transcript.

        This is metadata only (title, last-used) — message content already
        persists via the agent's checkpointer for as long as it is configured to
        keep it, which is also why a checkpointer is required to enable this:
        without one there is no transcript to read back. The default store for
        the metadata is in-memory and per-process; pass ``thread_store`` to the
        constructor for anything durable, such as ``SQLiteThreadStore``.
        """
        if enabled:
            self.adapter.require_checkpointer(reason="Thread history")
        self._threads = ThreadHistoryOptions(enabled=enabled)
        return self

    # ---- Building and serving --------------------------------------------

    def build(self, *, api_prefix: str = "/api") -> FastAPI:
        """Build the FastAPI application, serving the API and the UI.

        Built once and cached: repeated calls return the same object so ``run()``
        and a manual ``uvicorn`` invocation cannot diverge. ``api_prefix`` is
        therefore honored only on the first call.
        """
        if self._app is not None:
            return self._app

        if not api_prefix.startswith("/") or api_prefix.rstrip("/") == "":
            msg = f"api_prefix must be an absolute path such as '/api', got {api_prefix!r}."
            raise ConfigError(msg)
        api_prefix = api_prefix.rstrip("/")

        from fastapi import FastAPI
        from fastapi.responses import HTMLResponse
        from fastapi.staticfiles import StaticFiles

        app = FastAPI(title=self.title, docs_url="/api/docs", openapi_url="/api/openapi.json")
        app.include_router(self.router(), prefix=api_prefix)

        index = STATIC_DIR / "index.html"
        if not index.is_file():
            msg = (
                f"UI assets are missing from {STATIC_DIR}. The package may be installed "
                "incorrectly; reinstall agentstage."
            )
            raise ConfigError(msg)

        # index.html is served through a route rather than by StaticFiles so the API
        # prefix can be injected. The UI is served from "/" while the API lives
        # under a prefix, so relative fetch() URLs would resolve to the wrong path.
        html = index.read_text(encoding="utf-8").replace(
            'data-api-base="/api"', f'data-api-base="{api_prefix}"'
        )

        @app.get("/", include_in_schema=False)
        async def ui() -> HTMLResponse:
            return HTMLResponse(html)

        # Mounted last so it cannot shadow the API or the index route.
        app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="ui")

        self._app = app
        return app

    def mount(self, host_app: Any, *, prefix: str = "/agent", api_prefix: str = "/api") -> Self:
        """Mount the full app — API and chat UI — under ``prefix`` of an existing
        FastAPI application.

        Use this when a team already has a FastAPI service and wants the chat UI
        available at, say, ``/agent/`` alongside their own routes — as opposed to
        :meth:`router`, which mounts only the JSON API with no UI, for a team
        building their own frontend against it. The host's own routes, exception
        handlers, and middleware are untouched; agentstage owns nothing outside
        ``prefix``.
        """
        if not prefix.startswith("/") or prefix.rstrip("/") == "":
            msg = f"prefix must be an absolute path such as '/agent', got {prefix!r}."
            raise ConfigError(msg)
        host_app.mount(prefix.rstrip("/"), self.build(api_prefix=api_prefix))
        return self

    def router(self) -> Any:
        """The API router alone, for mounting into an existing FastAPI app."""
        attachments: AttachmentStore | None = None
        if self._chat.attachments:
            if self._attachment_store is None:
                self._attachment_store = InMemoryAttachmentStore()
            attachments = self._attachment_store

        threads: ThreadStore | None = None
        if self._threads.enabled:
            if self._thread_store is None:
                self._thread_store = InMemoryThreadStore()
            threads = self._thread_store

        return build_router(
            AppConfig(
                adapter=self.adapter,
                title=self.title,
                authenticate=self.authenticate,
                run_config=self.run_config,
                attachments=attachments,
                threads=threads,
            )
        )

    def run(self, *, host: str = "127.0.0.1", port: int = 8000) -> None:
        """Serve the app with uvicorn.

        Binds to localhost by default: a development default that listens on every
        interface is how an unauthenticated agent ends up exposed.
        """
        try:
            import uvicorn
        except ImportError as exc:  # pragma: no cover - depends on env
            msg = (
                "Running the server requires uvicorn, which is not installed. "
                "Install it with: pip install uvicorn"
            )
            raise ConfigError(msg) from exc

        print(f"agentstage — {self.title}")
        print(f"  UI:   http://{host}:{port}/")
        print(f"  API:  http://{host}:{port}/api/docs")
        uvicorn.run(self.build(), host=host, port=port)
