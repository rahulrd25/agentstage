"""LangGraph adapter — drives an agent and yields normalized events.

Named ``adapters/langgraph.py`` rather than ``langgraph/``: a package of that name
would shadow the real ``langgraph`` distribution and break imports.

The adapter owns run lifecycle and failure handling; :class:`StreamNormalizer`
owns event shape and correlation. Splitting them keeps the streaming logic
testable against canned chunks, with no agent involved.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator, Mapping
from typing import Any

from langgraph.types import Command

from agentstage.errors import AdapterError, ConfigError
from agentstage.events.models import AgentEvent
from agentstage.events.normalize import StreamNormalizer, citations_of, is_tool_message, text_of
from agentstage.types import SupportsAgentStream

__all__ = ["LangGraphAdapter"]

# Both channels are required: `messages` carries token deltas, `updates` carries
# tool lifecycle and interrupts. Neither alone is sufficient (verified).
_STREAM_MODES: tuple[str, ...] = ("updates", "messages")


class LangGraphAdapter:
    """Runs a compiled LangGraph agent and emits :class:`AgentEvent`s.

    The agent is validated at construction, so a graph that was never compiled
    fails immediately with a clear message instead of an ``AttributeError`` once a
    user is already waiting on a response.
    """

    def __init__(self, agent: SupportsAgentStream) -> None:
        if not hasattr(agent, "astream"):
            msg = (
                f"{type(agent).__name__} has no 'astream' method, so it cannot be streamed. "
                "If this is a StateGraph, call .compile() first: agent = graph.compile()."
            )
            raise AdapterError(msg)
        self.agent = agent

    # ---- Capability checks ------------------------------------------------

    def require_checkpointer(self, *, reason: str = "Human-in-the-loop approval") -> None:
        """Fail loudly if the graph has no checkpointer.

        Mitigates risk R3: LangGraph cannot interrupt without a checkpointer, and
        the symptom is a run that silently never pauses — one of the most common
        human-in-the-loop failures. Also the precondition for reading history
        back out of the checkpointer, hence the overridable ``reason`` — the fix
        is identical either way, but the consequence of skipping it is not.
        """
        checkpointer = getattr(self.agent, "checkpointer", None)
        if checkpointer is None:
            msg = (
                f"{reason} requires a checkpointer, but this agent was compiled without "
                "one. Fix it by compiling with a checkpointer, e.g.:\n\n"
                "    from langgraph.checkpoint.memory import InMemorySaver\n"
                "    agent = create_agent(model=model, tools=tools, checkpointer=InMemorySaver())"
            )
            raise ConfigError(msg)

    async def has_pending_interrupt(self, thread_id: str) -> bool:
        """Whether ``thread_id`` currently has a run paused on an interrupt.

        Verified against langgraph 1.2.11: resuming a thread with no pending
        interrupt does not error — ``interrupt()`` just re-runs the node from its
        start and produces a *new* interrupt. A stale or mistyped ``thread_id``
        would therefore silently start a fresh run instead of resuming the one the
        user meant to approve. Callers must check this before resuming.
        """
        snapshot = await self.agent.aget_state({"configurable": {"thread_id": thread_id}})
        return bool(snapshot.interrupts)

    async def pending_interrupt_value(self, thread_id: str) -> Any:
        """The value passed to the pending ``interrupt()`` call, or ``None``.

        ``has_pending_interrupt`` only reports existence; a caller building the
        resume payload needs to see the actual value to tell a hand-rolled
        ``interrupt(...)`` from a prebuilt middleware's own request shape (for
        example ``HumanInTheLoopMiddleware``, which expects a structured
        ``{"decisions": [...]}`` reply, not a bare value) — see
        ``runtime.fastapi._build_resume_value``.
        """
        snapshot = await self.agent.aget_state({"configurable": {"thread_id": thread_id}})
        if not snapshot.interrupts:
            return None
        return snapshot.interrupts[0].value

    async def get_history(self, thread_id: str) -> list[dict[str, Any]]:
        """Reconstruct a thread's transcript from the checkpointer's saved state.

        This does not duplicate storage: the checkpointer is already the source
        of truth for message content (see ``agentstage.storage``), so history is
        read from it directly rather than replayed from logged events. Reuses the
        same text/citation extraction the live event stream uses, so a message
        rendered from history looks identical to one rendered live.

        Returns ``[]`` for a thread the checkpointer has never seen — not an
        error, since a client listing an empty thread is a normal state, not a
        mistake. Requires a checkpointer at all — without one, LangGraph itself
        raises ``ValueError("No checkpointer set")`` on ``aget_state``, which this
        turns into the same actionable :class:`ConfigError` used elsewhere.
        """
        self.require_checkpointer(reason="Reading thread history")
        snapshot = await self.agent.aget_state({"configurable": {"thread_id": thread_id}})
        messages = snapshot.values.get("messages", []) if snapshot.values else []

        history: list[dict[str, Any]] = []
        for message in messages:
            if is_tool_message(message):
                history.append(
                    {
                        "role": "tool",
                        "tool_call_id": getattr(message, "tool_call_id", None),
                        "text": text_of(message),
                    }
                )
                continue

            role = "user" if getattr(message, "type", None) == "human" else "assistant"
            entry: dict[str, Any] = {"role": role, "text": text_of(message)}
            tool_calls = getattr(message, "tool_calls", None) or []
            if tool_calls:
                entry["tool_calls"] = [
                    {"id": call.get("id"), "name": call.get("name"), "args": call.get("args")}
                    for call in tool_calls
                ]
            citations = citations_of(message)
            if citations:
                entry["citations"] = citations
            history.append(entry)
        return history

    # ---- Streaming --------------------------------------------------------

    async def stream(
        self,
        message: str,
        *,
        thread_id: str | None = None,
        run_id: str | None = None,
        config: Mapping[str, Any] | None = None,
        content_blocks: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[AgentEvent]:
        """Run the agent on ``message``, yielding events as they occur.

        ``content_blocks`` are additional LangChain content blocks (e.g. from
        :func:`agentstage.files.content_block_for`) appended after the text block,
        so an uploaded image or file rides in the same ``HumanMessage`` as the
        prompt that references it.

        Every path terminates the stream with exactly one of ``run_completed`` or
        ``run_failed``, so a client always learns the run is over and can stop its
        spinner. Requirement 13: failures are reported, never swallowed.
        """
        content: Any = message
        if content_blocks:
            content = [{"type": "text", "text": message}, *content_blocks]
        graph_input: Any = {"messages": [("user", content)]}
        async for event in self._run(
            graph_input, thread_id=thread_id, run_id=run_id, config=config
        ):
            yield event

    async def resume(
        self,
        *,
        thread_id: str,
        resume_value: Any,
        run_id: str | None = None,
        config: Mapping[str, Any] | None = None,
    ) -> AsyncIterator[AgentEvent]:
        """Continue a run that is paused on an interrupt.

        Callers must check :meth:`has_pending_interrupt` first. Resuming a thread
        with no pending interrupt does not error in LangGraph — the interrupted
        node just re-runs from its start and produces a *new* interrupt (verified
        against 1.2.11) — so a stale ``thread_id`` would silently start an
        unrelated run instead of failing. This method re-checks and raises rather
        than trusting the caller did.
        """
        if not await self.has_pending_interrupt(thread_id):
            msg = (
                f"Thread {thread_id!r} has no pending interrupt to resume. Resuming it anyway "
                "would silently start a new run rather than continuing the approval the user "
                "meant to answer. It may have already been resumed, or never paused."
            )
            raise ConfigError(msg)

        async for event in self._run(
            Command(resume=resume_value), thread_id=thread_id, run_id=run_id, config=config
        ):
            yield event

    async def _run(
        self,
        graph_input: Any,
        *,
        thread_id: str | None,
        run_id: str | None,
        config: Mapping[str, Any] | None,
    ) -> AsyncIterator[AgentEvent]:
        run_id = run_id or f"run-{uuid.uuid4().hex[:12]}"
        normalizer = StreamNormalizer(run_id=run_id, thread_id=thread_id)

        for event in normalizer.run_started():
            yield event

        try:
            async for raw in self._raw_stream(graph_input, thread_id=thread_id, config=config):
                mode, payload = self._split_chunk(raw)
                for event in normalizer.handle(mode, payload):
                    yield event
        except Exception as exc:
            # A failing tool propagates here: LangGraph's default handler re-raises
            # anything that is not a ToolInvocationError (verified). Attribute it
            # to the open tool calls first, or their cards spin forever.
            for event in normalizer.tool_call_failed(exc):
                yield event
            for event in normalizer.run_failed(exc):
                yield event
            return

        if normalizer.interrupted:
            # A paused run is not a finished one. The interrupt_created event
            # already told the client to wait; emitting run_completed here would
            # tell it the opposite in the same breath.
            return

        for event in normalizer.run_completed():
            yield event

    def _raw_stream(
        self,
        graph_input: Any,
        *,
        thread_id: str | None,
        config: Mapping[str, Any] | None,
    ) -> AsyncIterator[Any]:
        merged = self._build_config(thread_id=thread_id, config=config)
        return self.agent.astream(
            graph_input,
            config=merged,
            stream_mode=list(_STREAM_MODES),
        )

    @staticmethod
    def _build_config(*, thread_id: str | None, config: Mapping[str, Any] | None) -> dict[str, Any]:
        """Merge a caller config with the thread_id LangGraph needs for state.

        The caller's ``configurable`` wins on conflict, except that an explicit
        ``thread_id`` argument is authoritative — it is what the UI resumes on.
        """
        merged: dict[str, Any] = dict(config or {})
        if thread_id is not None:
            configurable = dict(merged.get("configurable") or {})
            configurable["thread_id"] = thread_id
            merged["configurable"] = configurable
        return merged

    @staticmethod
    def _split_chunk(raw: Any) -> tuple[str, Any]:
        """Split a multi-mode chunk into ``(mode, payload)``.

        With multiple stream modes LangGraph yields 2-tuples; anything else means
        the contract shifted, so it is reported rather than guessed at.
        """
        if isinstance(raw, tuple) and len(raw) == 2:
            mode, payload = raw
            return str(mode), payload
        msg = (
            "Expected a (mode, payload) tuple from a multi-mode LangGraph stream, got "
            f"{type(raw).__name__}. The installed LangGraph may have changed its "
            "streaming contract."
        )
        raise AdapterError(msg)
