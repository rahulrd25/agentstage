"""Core type vocabulary for agentstage.

This module holds the type-level contract only: the literals, protocols, and
identifier types shared across the codebase. The ``AgentEvent`` model itself,
with validation and serialization, arrives in Milestone 2 under
``agentstage.events``.

Nothing here imports langgraph. Keeping this module dependency-free is what lets
the event contract stay stable while adapter backends are swapped underneath it.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, Literal, Protocol, runtime_checkable

__all__ = [
    "EventType",
    "MessageRole",
    "RunStatus",
    "StreamMode",
    "SupportsAgentStream",
    "ToolCallStatus",
]


EventType = Literal[
    "run_started",
    "run_completed",
    "run_failed",
    "message_started",
    "message_delta",
    "message_completed",
    "tool_call_started",
    "tool_call_delta",
    "tool_call_completed",
    "tool_call_failed",
    "interrupt_created",
    "progress_updated",
    "state_updated",
]
"""Normalized event types. This is the public contract the UI renders against."""


StreamMode = Literal[
    "values",
    "updates",
    "checkpoints",
    "tasks",
    "debug",
    "messages",
    "custom",
]
"""LangGraph stream channels.

Verified by introspection against langgraph 1.2.11 — this is the exact set, not
a superset from documentation. ``astream`` additionally accepts only
``version='v1'|'v2'``; ``v3`` exists solely on ``astream_events`` and is
experimental.
"""


RunStatus = Literal["pending", "running", "completed", "failed", "cancelled", "interrupted"]

ToolCallStatus = Literal["started", "running", "completed", "failed"]

MessageRole = Literal["user", "assistant", "tool", "system"]


@runtime_checkable
class SupportsAgentStream(Protocol):
    """The minimum surface an adapter needs from a compiled agent.

    Structural, not nominal: any object exposing ``astream`` qualifies, which
    keeps fake agents in tests on exactly the same code path as a real
    ``CompiledStateGraph``. Checked at configuration time so a graph that was
    never compiled fails with a clear :class:`~agentstage.errors.AdapterError`
    instead of an ``AttributeError`` mid-stream.

    ``astream`` is declared ``(*args, **kwargs)`` deliberately. A precise
    signature does *not* match the real ``CompiledStateGraph.astream``, whose
    ``config`` is a ``RunnableConfig`` and whose overloads are keyed on
    ``version`` — so a stricter protocol would reject the very type it exists to
    describe, and every caller passing a real agent would need a ``cast``. The
    adapter validates the call site itself; this protocol only promises the
    method exists.
    """

    def astream(self, *args: Any, **kwargs: Any) -> AsyncIterator[Any]:
        """Stream raw agent output."""
        ...

    def aget_state(self, *args: Any, **kwargs: Any) -> Any:
        """Fetch a thread's checkpoint state, used to detect a pending interrupt.

        Also loosely typed for the same reason as ``astream``: matching
        ``CompiledStateGraph.aget_state``'s real signature exactly would reject
        the type this protocol exists to describe.
        """
        ...
