"""agentstage — a Python-first UI layer for LangChain and LangGraph agents.

Example::

    from agentstage import AgentApp

    app = AgentApp(agent=agent, title="Research Assistant")
    app.chat(streaming=True)
    app.tool_calls()
    app.run()
"""

from typing import TYPE_CHECKING, Any

__version__ = "0.1.1"

__all__ = ["AgentApp", "AgentEvent", "__version__"]

if TYPE_CHECKING:  # pragma: no cover - for type checkers only
    from agentstage.app import AgentApp
    from agentstage.events import AgentEvent


def __getattr__(name: str) -> Any:
    """Resolve public names on first use.

    Importing ``agentstage`` must stay cheap: pulling in FastAPI and LangGraph at
    package import would make a library consumer pay for a server they may never
    start. The names are still importable directly, and type checkers see them via
    the ``TYPE_CHECKING`` block above.
    """
    if name == "AgentApp":
        from agentstage.app import AgentApp

        return AgentApp
    if name == "AgentEvent":
        from agentstage.events import AgentEvent

        return AgentEvent
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
