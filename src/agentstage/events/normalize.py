"""Normalize raw LangGraph stream output into :class:`AgentEvent`s.

The normalizer is a small state machine, not a pure function, because LangGraph
splits one logical conversation across channels that must be joined:

- ``messages`` carries token-level ``AIMessageChunk``s (and whole ``ToolMessage``s).
  Chunks sharing a ``message_id`` belong to one assistant message, so the first
  chunk emits ``message_started`` and later ones ``message_delta``.
- ``updates`` carries node-level state, where tool calls appear on
  ``AIMessage.tool_calls`` and results arrive later as separate ``ToolMessage``s.
  Started and completed calls are joined on ``tool_call_id``.

All shapes below were verified by introspection against langgraph 1.2.11; see
``tests/unit/test_normalize.py`` for the pinned expectations.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from agentstage.errors import EventNormalizationError
from agentstage.events.models import AgentEvent

__all__ = ["StreamNormalizer", "citations_of", "is_tool_message", "text_of"]

# Keys copied from LangGraph's `messages` metadata onto emitted events. An
# allowlist, not a denylist: provider metadata is untrusted and may carry prompts
# or reasoning, so unknown keys are dropped rather than forwarded.
_SAFE_METADATA_KEYS: frozenset[str] = frozenset(
    {"langgraph_node", "langgraph_step", "thread_id", "checkpoint_ns"}
)

# LangGraph reports interrupts under this reserved key in the `updates` channel.
_INTERRUPT_KEY = "__interrupt__"


class StreamNormalizer:
    """Turns raw stream chunks into ordered, correlated :class:`AgentEvent`s.

    One instance per run: it holds the per-run correlation state (which messages
    have started, which tool calls are open) and stamps a monotonic ``sequence``
    so a client can detect gaps and order events it received out of order.

    Usage is one call per raw chunk, each yielding zero or more events::

        norm = StreamNormalizer(run_id="r1", thread_id="t1")
        for event in norm.run_started():
            ...
        async for mode, payload in agent.astream(..., stream_mode=["updates", "messages"]):
            for event in norm.handle(mode, payload):
                ...
    """

    def __init__(self, run_id: str, thread_id: str | None = None) -> None:
        if not run_id:
            msg = "StreamNormalizer requires a non-empty run_id."
            raise EventNormalizationError(msg)
        self.run_id = run_id
        self.thread_id = thread_id
        self._sequence = 0
        #: Whether this run has emitted an interrupt. A paused run is not a
        #: finished one — the adapter uses this to skip `run_completed`.
        self.interrupted = False
        #: message ids that have already emitted `message_started`
        self._started_messages: set[str] = set()
        #: message ids that have already emitted `message_completed` at least
        #: once — a later node revising the same id (redaction, citation
        #: injection, or any other `after_model`-style middleware) still needs
        #: its own `message_completed`, not silent ignoring, see `_finish_message`.
        self._completed_messages: set[str] = set()
        #: tool_call_id -> tool name, for calls awaiting a result
        self._open_tool_calls: dict[str, str] = {}
        #: tool_call_ids already completed or failed, to drop duplicate results
        self._closed_tool_calls: set[str] = set()

    # ---- Lifecycle --------------------------------------------------------

    def run_started(self) -> list[AgentEvent]:
        return [self._build(AgentEvent.run_started)]

    def run_completed(self) -> list[AgentEvent]:
        """Close the run, completing any message left open by the stream.

        A stream can end without a terminal chunk for the last message; emitting
        the missing ``message_completed`` keeps the UI from showing a message
        that streams forever.
        """
        events = [
            self._stamp(self._message_completed(message_id))
            for message_id in sorted(self._started_messages)
        ]
        self._started_messages.clear()
        events.append(self._build(AgentEvent.run_completed))
        return events

    def run_failed(self, error: BaseException) -> list[AgentEvent]:
        """Report a run failure.

        Only the exception's message and class name cross the boundary — the
        traceback stays server-side.
        """
        return [
            self._build(
                AgentEvent.run_failed,
                error=str(error) or error.__class__.__name__,
                error_type=type(error).__name__,
            )
        ]

    def tool_call_failed(self, error: BaseException) -> list[AgentEvent]:
        """Attribute a propagating exception to the tool calls still open.

        LangGraph's default ``handle_tool_errors`` re-raises anything that is not
        a ``ToolInvocationError`` (verified), so a failing tool surfaces as an
        exception rather than a ``ToolMessage``. Without this, the UI would leave
        the tool card spinning forever and only show a run-level failure.
        """
        events: list[AgentEvent] = []
        for tool_call_id, name in list(self._open_tool_calls.items()):
            events.append(
                self._build(
                    AgentEvent.tool_call_failed,
                    tool_call_id=tool_call_id,
                    name=name,
                    error=str(error) or error.__class__.__name__,
                )
            )
            self._closed_tool_calls.add(tool_call_id)
        self._open_tool_calls.clear()
        return events

    # ---- Dispatch ---------------------------------------------------------

    def handle(self, mode: str, payload: Any) -> list[AgentEvent]:
        """Normalize one raw chunk from a given stream mode.

        Unrecognized modes yield nothing rather than raising: a future LangGraph
        adding a channel should not break a running app. Malformed payloads
        *within* a known mode do raise, because that is a real contract mismatch.
        """
        if mode == "messages":
            return self._handle_messages(payload)
        if mode == "updates":
            return self._handle_updates(payload)
        return []

    # ---- messages channel -------------------------------------------------

    def _handle_messages(self, payload: Any) -> list[AgentEvent]:
        """Handle a ``(message, metadata)`` pair from the ``messages`` channel."""
        if not isinstance(payload, tuple) or len(payload) != 2:
            msg = (
                "The 'messages' stream mode must yield a (message, metadata) tuple; got "
                f"{type(payload).__name__}. This usually means the installed LangGraph "
                "changed its stream contract."
            )
            raise EventNormalizationError(msg)

        message, raw_metadata = payload
        metadata = _safe_metadata(raw_metadata)
        node_name = metadata.get("langgraph_node")

        # ToolMessages arrive here too, but tool lifecycle is driven from the
        # `updates` channel where the originating call is visible. Handling both
        # would double-emit.
        if is_tool_message(message):
            return []

        text = text_of(message)
        message_id = getattr(message, "id", None)
        if not message_id:
            # Without an id, deltas cannot be appended to the right message.
            return []

        events: list[AgentEvent] = []
        if message_id not in self._started_messages:
            self._started_messages.add(message_id)
            events.append(
                self._stamp(
                    AgentEvent(
                        type="message_started",
                        run_id=self.run_id,
                        thread_id=self.thread_id,
                        message_id=message_id,
                        node_name=node_name,
                    )
                )
            )

        # An empty chunk is the tool-call-only message: it opens the message but
        # carries no text to append.
        if text:
            events.append(
                self._build(
                    AgentEvent.message_delta,
                    message_id=message_id,
                    text=text,
                    node_name=node_name,
                )
            )
        return events

    # ---- updates channel --------------------------------------------------

    def _handle_updates(self, payload: Any) -> list[AgentEvent]:
        """Handle a ``{node_name: state}`` dict from the ``updates`` channel."""
        if not isinstance(payload, dict):
            msg = (
                "The 'updates' stream mode must yield a {node_name: state} dict; got "
                f"{type(payload).__name__}."
            )
            raise EventNormalizationError(msg)

        events: list[AgentEvent] = []
        for node_name, node_state in payload.items():
            if node_name == _INTERRUPT_KEY:
                events.extend(self._handle_interrupt(node_state))
                continue
            events.extend(self._handle_node_update(str(node_name), node_state))
        return events

    def _handle_node_update(self, node_name: str, node_state: Any) -> list[AgentEvent]:
        messages = _messages_of(node_state)
        events: list[AgentEvent] = []
        for message in messages:
            if is_tool_message(message):
                events.extend(self._complete_tool_call(message, node_name))
            else:
                events.extend(self._start_tool_calls(message, node_name))
                events.extend(self._finish_message(message))
        return events

    def _start_tool_calls(self, message: Any, node_name: str) -> list[AgentEvent]:
        """Emit ``tool_call_started`` for each call on an assistant message.

        Parallel calls arrive together on one message but their results come back
        as separate updates, in no guaranteed order — hence correlation by id.
        """
        events: list[AgentEvent] = []
        for call in getattr(message, "tool_calls", None) or []:
            tool_call_id = call.get("id")
            if not tool_call_id:
                # `ToolCall["id"]` is typed `str | None`. A call with no id cannot
                # be correlated to its result, so skip rather than emit a card
                # that can never complete.
                continue
            if tool_call_id in self._open_tool_calls or tool_call_id in self._closed_tool_calls:
                continue
            name = call.get("name") or "unknown_tool"
            self._open_tool_calls[tool_call_id] = name
            events.append(
                self._build(
                    AgentEvent.tool_call_started,
                    tool_call_id=tool_call_id,
                    name=name,
                    args=call.get("args") or {},
                    node_name=node_name,
                )
            )
        return events

    def _complete_tool_call(self, message: Any, node_name: str) -> list[AgentEvent]:
        """Emit completion (or failure) for a ``ToolMessage``."""
        tool_call_id = getattr(message, "tool_call_id", None)
        if not tool_call_id or tool_call_id in self._closed_tool_calls:
            return []

        name = self._open_tool_calls.pop(tool_call_id, None)
        self._closed_tool_calls.add(tool_call_id)

        # An app that configures `handle_tool_errors` gets a ToolMessage with
        # status='error' instead of a propagating exception. Both must render as
        # a failed tool call.
        if getattr(message, "status", None) == "error":
            return [
                self._build(
                    AgentEvent.tool_call_failed,
                    tool_call_id=tool_call_id,
                    error=text_of(message) or "The tool reported an error.",
                    name=name,
                    node_name=node_name,
                )
            ]
        return [
            self._build(
                AgentEvent.tool_call_completed,
                tool_call_id=tool_call_id,
                result=text_of(message),
                name=name,
                node_name=node_name,
            )
        ]

    def _finish_message(self, message: Any) -> list[AgentEvent]:
        """Close a message once its final form appears on ``updates``.

        A node later than the one that first produced a message can still
        revise it under the same id — a redaction, citation-injection, or any
        other ``after_model``-style middleware editing the model's own reply.
        That revision must still reach the client as its own
        ``message_completed``: the frontend already upserts by ``message_id``,
        so re-emitting here is what lets it show the corrected content instead
        of the stale pre-middleware text forever (verified: the checkpointer's
        own final state, and therefore ``get_history()``, already reflects the
        revision — without this, a live run and a reload of the same thread
        permanently disagree on what the agent actually said).
        """
        message_id = getattr(message, "id", None)
        if not message_id:
            return []
        if message_id in self._started_messages:
            self._started_messages.discard(message_id)
            self._completed_messages.add(message_id)
            return [self._stamp(self._message_completed(message_id, message))]
        if message_id in self._completed_messages:
            return [self._stamp(self._message_completed(message_id, message))]
        return []

    def _handle_interrupt(self, raw: Any) -> list[AgentEvent]:
        """Emit an event per pending interrupt.

        ``__interrupt__`` holds a *tuple* of ``Interrupt`` objects, each with
        exactly ``value`` and ``id`` (verified against 1.2.11).
        """
        interrupts = raw if isinstance(raw, tuple | list) else [raw]
        events: list[AgentEvent] = []
        for item in interrupts:
            interrupt_id = getattr(item, "id", None)
            if not interrupt_id:
                continue
            self.interrupted = True
            events.append(
                self._build(
                    AgentEvent.interrupt_created,
                    interrupt_id=str(interrupt_id),
                    value=getattr(item, "value", None),
                )
            )
        return events

    # ---- Helpers ----------------------------------------------------------

    def _message_completed(self, message_id: str, message: Any = None) -> AgentEvent:
        data: dict[str, Any] | None = None
        if message is not None:
            text = text_of(message)
            citations = citations_of(message)
            if text or citations:
                data = {}
                if text:
                    data["text"] = text
                if citations:
                    # Citations are metadata on the final message, not something
                    # that streams token-by-token, so they only ever appear here
                    # — never on a message_delta.
                    data["citations"] = citations
        return AgentEvent(
            type="message_completed",
            run_id=self.run_id,
            thread_id=self.thread_id,
            message_id=message_id,
            data=data,
        )

    def _stamp(self, event: AgentEvent) -> AgentEvent:
        """Assign the next sequence number to an event.

        Sequencing lives in one place so no call site can forget it; a client uses
        the numbers to detect dropped SSE frames.
        """
        stamped = replace(event, sequence=self._sequence)
        self._sequence += 1
        return stamped

    def _build(self, factory: Any, **kwargs: Any) -> AgentEvent:
        """Construct via an :class:`AgentEvent` classmethod, then stamp it."""
        return self._stamp(factory(self.run_id, thread_id=self.thread_id, **kwargs))


def _safe_metadata(raw: Any) -> dict[str, Any]:
    """Copy only known-safe metadata keys.

    Provider metadata is untrusted and can carry prompts or reasoning, so this is
    an allowlist.
    """
    if not isinstance(raw, dict):
        return {}
    return {k: v for k, v in raw.items() if k in _SAFE_METADATA_KEYS}


def is_tool_message(message: Any) -> bool:
    """Identify a ToolMessage structurally, without importing langchain here."""
    return getattr(message, "type", None) == "tool" or hasattr(message, "tool_call_id")


def text_of(message: Any) -> str:
    """Extract plain text from a message whose content may be a list of blocks.

    Multimodal content arrives as a list of dicts; only text blocks are joined.
    Non-text blocks are dropped rather than stringified into noise.
    """
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return "".join(parts)
    return str(content) if content else ""


def citations_of(message: Any) -> list[dict[str, Any]]:
    """Extract citations from a message's content blocks, if any.

    Verified against langchain-core 1.5.4: a cited response arrives as
    ``content = [{"type": "text", "text": ..., "annotations": [Citation, ...]}]``
    — a standard shape (``langchain_core.messages.content.Citation``), not
    something agentstage invents. Only the fields the UI needs are copied, so an
    unrecognized future annotation type or an unexpected extra key does not leak
    through unfiltered.
    """
    content = getattr(message, "content", None)
    if not isinstance(content, list):
        return []
    citations: list[dict[str, Any]] = []
    for block in content:
        if not isinstance(block, dict) or block.get("type") != "text":
            continue
        for annotation in block.get("annotations") or []:
            if not isinstance(annotation, dict) or annotation.get("type") != "citation":
                continue
            citation = {
                k: annotation[k]
                for k in ("url", "title", "cited_text", "start_index", "end_index")
                if annotation.get(k) is not None
            }
            if citation:
                citations.append(citation)
    return citations


def _messages_of(node_state: Any) -> list[Any]:
    """Pull the message list out of a node's state update, tolerating shapes.

    A node may return no messages at all (a state-only update), or a single
    message rather than a list.
    """
    if not isinstance(node_state, dict):
        return []
    messages = node_state.get("messages")
    if messages is None:
        return []
    if isinstance(messages, list):
        return messages
    return [messages]
