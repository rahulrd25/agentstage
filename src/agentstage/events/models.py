"""The normalized event model.

``AgentEvent`` is the stable contract between adapters and the UI. Adapter
backends may be swapped — stable stream modes today, ``astream_events(v3)`` when
it leaves experimental — without any change here or in the UI.

Design notes:

- Every field except ``type`` and ``run_id`` is optional. Real LangGraph events do
  not all carry a node name, a message id, or a thread id, and inventing values
  to fill a rigid schema would be a lie the UI then renders.
- ``data`` is the typed-ish payload the UI reads; ``metadata`` preserves useful
  original context. Neither may carry chain-of-thought (see ``sanitized``).
- Events are frozen. They are a record of something that already happened, and
  the SSE transport may hand the same event to several consumers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from typing import Any, ClassVar, Self

from agentstage.errors import EventNormalizationError
from agentstage.types import EventType

__all__ = ["EVENT_TYPES", "AgentEvent"]


EVENT_TYPES: frozenset[str] = frozenset(
    {
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
    }
)
"""Runtime-checkable mirror of :data:`~agentstage.types.EventType`.

``Literal`` is erased at runtime, so validation needs a real set. The contract
test asserts the two never drift apart.
"""


# Keys that must never reach the browser. Reasoning content is the security
# requirement (no chain-of-thought exposure); the rest are provider internals
# that leak prompts or raw model output.
_FORBIDDEN_DATA_KEYS: frozenset[str] = frozenset(
    {
        "reasoning",
        "reasoning_content",
        "thinking",
        "thought",
        "chain_of_thought",
        "additional_kwargs",
    }
)


@dataclass(frozen=True, slots=True)
class AgentEvent:
    """A single normalized agent event.

    Construct through the classmethods where possible — they enforce the
    field combinations each event type requires.
    """

    type: EventType
    run_id: str
    thread_id: str | None = None
    node_name: str | None = None
    message_id: str | None = None
    tool_call_id: str | None = None
    sequence: int | None = None
    data: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None

    #: Fields carried through serialization, in a stable order.
    FIELDS: ClassVar[tuple[str, ...]] = (
        "type",
        "run_id",
        "thread_id",
        "node_name",
        "message_id",
        "tool_call_id",
        "sequence",
        "data",
        "metadata",
    )

    def __post_init__(self) -> None:
        if self.type not in EVENT_TYPES:
            known = ", ".join(sorted(EVENT_TYPES))
            msg = f"Unknown event type {self.type!r}. Expected one of: {known}."
            raise EventNormalizationError(msg)

        if not self.run_id:
            msg = (
                f"Event {self.type!r} has an empty run_id. Every event must carry a run_id "
                "so the UI can correlate it to a run; generate one when the run starts."
            )
            raise EventNormalizationError(msg)

        if self.sequence is not None and self.sequence < 0:
            msg = f"Event sequence must be non-negative, got {self.sequence}."
            raise EventNormalizationError(msg)

        if self.type.startswith("tool_call_") and not self.tool_call_id:
            msg = (
                f"Event {self.type!r} requires a tool_call_id. Tool lifecycle events are "
                "correlated by it — a card cannot be matched to its result without one."
            )
            raise EventNormalizationError(msg)

        if self.type.startswith("message_") and not self.message_id:
            msg = (
                f"Event {self.type!r} requires a message_id, so streamed deltas append to "
                "the right message instead of creating a new one per token."
            )
            raise EventNormalizationError(msg)

    # ---- Safety -----------------------------------------------------------

    @property
    def sanitized(self) -> Self:
        """A copy with reasoning and provider internals stripped from ``data``.

        Applied at the transport boundary, not at construction: server-side
        logging and debugging legitimately want the full payload, while the
        browser must never receive chain-of-thought.
        """
        if not self.data:
            return self
        clean = {k: v for k, v in self.data.items() if k not in _FORBIDDEN_DATA_KEYS}
        if len(clean) == len(self.data):
            return self
        return replace(self, data=clean)

    # ---- Serialization ----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize, omitting keys that are unset.

        Dropping ``None`` keeps SSE frames small — the payload is sent per token,
        so absent fields should not cost bytes.
        """
        out: dict[str, Any] = {}
        for name in self.FIELDS:
            value = getattr(self, name)
            if value is not None:
                out[name] = value
        return out

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Self:
        """Rebuild from :meth:`to_dict` output, rejecting unknown keys."""
        unknown = set(payload) - set(cls.FIELDS)
        if unknown:
            msg = (
                f"Unknown event field(s): {', '.join(sorted(unknown))}. "
                f"Known fields: {', '.join(cls.FIELDS)}."
            )
            raise EventNormalizationError(msg)
        if "type" not in payload:
            msg = "Event payload is missing the required 'type' field."
            raise EventNormalizationError(msg)
        if "run_id" not in payload:
            msg = "Event payload is missing the required 'run_id' field."
            raise EventNormalizationError(msg)
        return cls(**payload)

    def to_json(self) -> str:
        """Serialize to JSON for the SSE ``data:`` field.

        Non-JSON-native values are coerced with ``default=str`` rather than
        raising: a tool returning an arbitrary object should degrade to a
        readable string, not kill an in-flight run.
        """
        return json.dumps(self.to_dict(), default=str, ensure_ascii=False)

    @classmethod
    def from_json(cls, raw: str) -> Self:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            msg = f"Event payload is not valid JSON: {exc}"
            raise EventNormalizationError(msg) from exc
        if not isinstance(payload, dict):
            msg = f"Event payload must be a JSON object, got {type(payload).__name__}."
            raise EventNormalizationError(msg)
        return cls.from_dict(payload)

    # ---- Debugging --------------------------------------------------------

    def describe(self) -> str:
        """One-line human-readable form, for logs and troubleshooting.

        Deliberately not ``__repr__``: the dataclass repr stays complete for
        test failure output, while this stays scannable in a stream of events.
        """
        parts: list[str] = [self.type]
        if self.node_name:
            parts.append(f"node={self.node_name}")
        if self.tool_call_id:
            parts.append(f"tool={self.tool_call_id}")
        if self.message_id:
            parts.append(f"msg={_shorten(self.message_id)}")
        if self.data:
            preview = ", ".join(f"{k}={_preview(v)}" for k, v in self.data.items())
            parts.append(f"data({preview})")
        seq = "" if self.sequence is None else f"#{self.sequence} "
        return f"[{seq}run={_shorten(self.run_id)}] " + " ".join(parts)

    # ---- Constructors -----------------------------------------------------

    @classmethod
    def run_started(cls, run_id: str, *, thread_id: str | None = None, **extra: Any) -> Self:
        return cls(type="run_started", run_id=run_id, thread_id=thread_id, **extra)

    @classmethod
    def run_completed(cls, run_id: str, *, thread_id: str | None = None, **extra: Any) -> Self:
        return cls(type="run_completed", run_id=run_id, thread_id=thread_id, **extra)

    @classmethod
    def run_failed(
        cls,
        run_id: str,
        *,
        error: str,
        error_type: str | None = None,
        thread_id: str | None = None,
        **extra: Any,
    ) -> Self:
        """Build a failure event.

        ``error`` is a message, never an exception object: the traceback stays
        server-side, and requirement 13 forbids swallowing the failure silently.
        """
        data: dict[str, Any] = {"error": error}
        if error_type:
            data["error_type"] = error_type
        return cls(type="run_failed", run_id=run_id, thread_id=thread_id, data=data, **extra)

    @classmethod
    def message_delta(
        cls,
        run_id: str,
        *,
        message_id: str,
        text: str,
        node_name: str | None = None,
        **extra: Any,
    ) -> Self:
        """An incremental chunk of assistant text.

        Only the delta travels, never the accumulated message — sending the whole
        message per token is the performance failure this model exists to avoid.
        """
        return cls(
            type="message_delta",
            run_id=run_id,
            message_id=message_id,
            node_name=node_name,
            data={"text": text},
            **extra,
        )

    @classmethod
    def tool_call_started(
        cls,
        run_id: str,
        *,
        tool_call_id: str,
        name: str,
        args: dict[str, Any] | None = None,
        node_name: str | None = None,
        **extra: Any,
    ) -> Self:
        return cls(
            type="tool_call_started",
            run_id=run_id,
            tool_call_id=tool_call_id,
            node_name=node_name,
            data={"name": name, "args": args or {}},
            **extra,
        )

    @classmethod
    def tool_call_completed(
        cls,
        run_id: str,
        *,
        tool_call_id: str,
        result: Any,
        name: str | None = None,
        node_name: str | None = None,
        **extra: Any,
    ) -> Self:
        data: dict[str, Any] = {"result": result}
        if name:
            data["name"] = name
        return cls(
            type="tool_call_completed",
            run_id=run_id,
            tool_call_id=tool_call_id,
            node_name=node_name,
            data=data,
            **extra,
        )

    @classmethod
    def tool_call_failed(
        cls,
        run_id: str,
        *,
        tool_call_id: str,
        error: str,
        name: str | None = None,
        node_name: str | None = None,
        **extra: Any,
    ) -> Self:
        data: dict[str, Any] = {"error": error}
        if name:
            data["name"] = name
        return cls(
            type="tool_call_failed",
            run_id=run_id,
            tool_call_id=tool_call_id,
            node_name=node_name,
            data=data,
            **extra,
        )

    @classmethod
    def interrupt_created(
        cls,
        run_id: str,
        *,
        interrupt_id: str,
        value: Any,
        thread_id: str | None = None,
        **extra: Any,
    ) -> Self:
        """A human-in-the-loop interrupt.

        ``Interrupt`` carries exactly ``value`` and ``id`` in langgraph 1.2.11 —
        verified, not assumed. Resuming requires the thread_id, so a missing one
        here means the run cannot be resumed.
        """
        return cls(
            type="interrupt_created",
            run_id=run_id,
            thread_id=thread_id,
            data={"interrupt_id": interrupt_id, "value": value},
            **extra,
        )


def _shorten(value: str, keep: int = 8) -> str:
    return value if len(value) <= keep else f"{value[:keep]}…"


def _preview(value: Any, limit: int = 40) -> str:
    """Render a value on one line, keeping whitespace-only values visible.

    Newlines and tabs are escaped rather than collapsed: a streamed token can be a
    single space, and showing that as an empty string makes a correct stream look
    broken in the logs.
    """
    text = str(value)
    # repr() escapes newlines and tabs and quotes the result, so leading, trailing,
    # or whitespace-only content stays visible.
    rendered = repr(text) if text != text.strip() or "\n" in text or "\t" in text else text
    if len(rendered) > limit:
        return f"{rendered[:limit]}…"
    return rendered
