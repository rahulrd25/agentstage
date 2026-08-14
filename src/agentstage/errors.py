"""Exception hierarchy for agentstage.

Every error carries an actionable message: what went wrong, and what to change.
Callers can catch :class:`AgentStageError` to trap anything raised by this library.
"""

from __future__ import annotations


class AgentStageError(Exception):
    """Base class for every error raised by agentstage."""


class ConfigError(AgentStageError):
    """Invalid or inconsistent configuration, detected before a run starts.

    Raised at configuration time rather than mid-run so the developer sees the
    problem while wiring the app up, not as a hang or a silent no-op in
    production. The canonical case is enabling human-in-the-loop approval on a
    graph compiled without a checkpointer, which LangGraph would otherwise fail
    to interrupt.
    """


class AdapterError(AgentStageError):
    """The agent object is not something an adapter can drive.

    Raised when the supplied object does not expose the streaming interface an
    adapter requires — for example, a bare graph that was never compiled.
    """


class EventNormalizationError(AgentStageError):
    """A raw agent event could not be normalized into an ``AgentEvent``.

    Signals a real mismatch between the installed LangGraph version and the
    adapter's expectations. It is deliberately loud: silently dropping events
    would show the user an incomplete conversation with no indication anything
    was lost.
    """


class AgentRunError(AgentStageError):
    """The underlying agent raised while a run was in flight.

    Wraps the original exception as ``__cause__`` so the agent's own traceback
    survives. This is a genuine failure of the user's agent, not of agentstage.
    """


class RunCancelledError(AgentStageError):
    """A run was cancelled before it completed.

    Cancellation is an expected outcome, not a defect: it is what a user
    pressing "stop" produces. It is an exception only so in-flight streaming
    unwinds through the same path as a failure.
    """
