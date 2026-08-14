"""Normalizer tests against canned chunks — no agent, no event loop.

The shapes here were captured by introspecting langgraph 1.2.11, so these tests
pin the real contract rather than a guess at it.
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage

from agentstage.errors import EventNormalizationError
from agentstage.events import AgentEvent, StreamNormalizer

MESSAGES_META = {"langgraph_node": "model", "thread_id": "t1", "langgraph_step": 1}


def types_of(events: list[AgentEvent]) -> list[str]:
    return [e.type for e in events]


def new_normalizer() -> StreamNormalizer:
    return StreamNormalizer(run_id="r1", thread_id="t1")


# ---- Construction --------------------------------------------------------


def test_requires_a_run_id():
    with pytest.raises(EventNormalizationError, match="non-empty run_id"):
        StreamNormalizer(run_id="")


# ---- Sequencing ----------------------------------------------------------


def test_sequence_is_monotonic_across_channels():
    """A client uses sequence numbers to detect dropped SSE frames, so they must
    increase across the whole run, not per channel."""
    norm = new_normalizer()

    events = norm.run_started()
    events += norm.handle("messages", (AIMessageChunk(content="hi", id="m1"), MESSAGES_META))
    events += norm.handle("updates", {"model": {"messages": [AIMessage(content="hi", id="m1")]}})
    events += norm.run_completed()

    assert [e.sequence for e in events] == list(range(len(events)))


def test_first_event_is_sequence_zero():
    assert new_normalizer().run_started()[0].sequence == 0


# ---- messages channel ----------------------------------------------------


def test_first_chunk_starts_a_message_then_deltas_follow():
    """Chunks share an id, so only the first opens the message."""
    norm = new_normalizer()

    first = norm.handle("messages", (AIMessageChunk(content="Hel", id="m1"), MESSAGES_META))
    second = norm.handle("messages", (AIMessageChunk(content="lo", id="m1"), MESSAGES_META))

    assert types_of(first) == ["message_started", "message_delta"]
    assert types_of(second) == ["message_delta"]
    assert [e.data for e in second] == [{"text": "lo"}]


def test_only_the_delta_is_sent_never_the_accumulation():
    """Performance requirement: sending the whole message per token is the failure
    this model exists to prevent."""
    norm = new_normalizer()
    norm.handle("messages", (AIMessageChunk(content="Hel", id="m1"), MESSAGES_META))

    events = norm.handle("messages", (AIMessageChunk(content="lo", id="m1"), MESSAGES_META))

    assert events[0].data == {"text": "lo"}, "must be the delta, not 'Hello'"


def test_node_name_is_carried_from_metadata():
    norm = new_normalizer()

    events = norm.handle("messages", (AIMessageChunk(content="x", id="m1"), MESSAGES_META))

    assert all(e.node_name == "model" for e in events)


def test_empty_chunk_opens_the_message_but_emits_no_delta():
    """A tool-call-only message has empty content; a delta of '' would render as a
    blank bubble."""
    norm = new_normalizer()

    events = norm.handle("messages", (AIMessageChunk(content="", id="m1"), MESSAGES_META))

    assert types_of(events) == ["message_started"]


def test_a_message_without_an_id_is_skipped():
    """Deltas cannot be routed without an id; emitting them would create a new
    bubble per token."""
    norm = new_normalizer()

    assert norm.handle("messages", (AIMessageChunk(content="x", id=None), MESSAGES_META)) == []


def test_tool_messages_on_the_messages_channel_are_ignored():
    """Tool lifecycle is driven from `updates`, where the originating call is
    visible. Handling both channels would double-emit."""
    norm = new_normalizer()
    tool_msg = ToolMessage(content="RESULT", tool_call_id="c1", id="tm1")

    assert norm.handle("messages", (tool_msg, MESSAGES_META)) == []


def test_multimodal_content_blocks_are_reduced_to_text():
    norm = new_normalizer()
    message = AIMessageChunk(
        content=[{"type": "text", "text": "hello "}, {"type": "image_url", "image_url": "..."}],
        id="m1",
    )

    events = norm.handle("messages", (message, MESSAGES_META))

    assert events[-1].data == {"text": "hello "}, "non-text blocks must be dropped, not stringified"


def test_malformed_messages_payload_is_reported():
    """A real contract mismatch must be loud, not silently skipped."""
    with pytest.raises(EventNormalizationError, match="must yield a \\(message, metadata\\) tuple"):
        new_normalizer().handle("messages", {"not": "a tuple"})


def test_unsafe_metadata_keys_are_not_copied_onto_events():
    """Provider metadata is untrusted and may carry prompts; only an allowlist is
    forwarded."""
    norm = new_normalizer()
    meta = {**MESSAGES_META, "ls_provider": "secret", "prompt": "leak me"}

    events = norm.handle("messages", (AIMessageChunk(content="x", id="m1"), meta))

    for event in events:
        assert event.metadata is None or "prompt" not in event.metadata


# ---- updates channel: tool lifecycle ------------------------------------


def _ai_with_calls(*calls: dict[str, Any]) -> AIMessage:
    return AIMessage(content="", tool_calls=list(calls), id="m1")


def _call(name: str, args: dict[str, Any], call_id: str) -> dict[str, Any]:
    return {"name": name, "args": args, "id": call_id, "type": "tool_call"}


def test_tool_call_started_comes_from_the_updates_channel():
    """The Milestone 0 finding: tool calls appear on AIMessage.tool_calls in
    `updates`, never on `messages`."""
    norm = new_normalizer()

    events = norm.handle(
        "updates", {"model": {"messages": [_ai_with_calls(_call("search", {"q": "x"}, "c1"))]}}
    )

    assert types_of(events) == ["tool_call_started"]
    assert events[0].tool_call_id == "c1"
    assert events[0].data == {"name": "search", "args": {"q": "x"}}
    assert events[0].node_name == "model"


def test_tool_result_correlates_by_tool_call_id():
    norm = new_normalizer()
    norm.handle(
        "updates", {"model": {"messages": [_ai_with_calls(_call("search", {"q": "x"}, "c1"))]}}
    )

    events = norm.handle(
        "updates", {"tools": {"messages": [ToolMessage(content="RESULT", tool_call_id="c1")]}}
    )

    assert types_of(events) == ["tool_call_completed"]
    assert events[0].tool_call_id == "c1"
    assert events[0].data == {"result": "RESULT", "name": "search"}


def test_parallel_calls_correlate_independently_and_out_of_order():
    """Results arrive as separate updates with no guaranteed order (verified), so
    correlation must be by id, not by position."""
    norm = new_normalizer()
    norm.handle(
        "updates",
        {
            "model": {
                "messages": [
                    _ai_with_calls(_call("t", {"q": "a"}, "c_a"), _call("t", {"q": "b"}, "c_b"))
                ]
            }
        },
    )

    second = norm.handle(
        "updates", {"tools": {"messages": [ToolMessage(content="B", tool_call_id="c_b")]}}
    )
    first = norm.handle(
        "updates", {"tools": {"messages": [ToolMessage(content="A", tool_call_id="c_a")]}}
    )

    assert second[0].tool_call_id == "c_b"
    assert second[0].data is not None and second[0].data["result"] == "B"
    assert first[0].tool_call_id == "c_a"
    assert first[0].data is not None and first[0].data["result"] == "A"


def test_a_tool_call_without_an_id_is_skipped():
    """`ToolCall['id']` is typed `str | None`. A card that can never complete is
    worse than no card."""
    norm = new_normalizer()

    events = norm.handle(
        "updates", {"model": {"messages": [_ai_with_calls(_call("search", {}, ""))]}}
    )

    assert events == []


def test_a_duplicate_started_call_is_not_re_emitted():
    """The same AIMessage can reappear across updates; the UI must not stack
    duplicate cards."""
    norm = new_normalizer()
    payload = {"model": {"messages": [_ai_with_calls(_call("search", {}, "c1"))]}}

    assert types_of(norm.handle("updates", payload)) == ["tool_call_started"]
    assert norm.handle("updates", payload) == []


def test_a_duplicate_tool_result_is_ignored():
    norm = new_normalizer()
    norm.handle("updates", {"model": {"messages": [_ai_with_calls(_call("s", {}, "c1"))]}})
    result = {"tools": {"messages": [ToolMessage(content="R", tool_call_id="c1")]}}

    assert types_of(norm.handle("updates", result)) == ["tool_call_completed"]
    assert norm.handle("updates", result) == []


def test_tool_message_with_error_status_becomes_a_failure():
    """An app configuring `handle_tool_errors` gets ToolMessage(status='error')
    instead of a raise. Both must render as a failed call."""
    norm = new_normalizer()
    norm.handle("updates", {"model": {"messages": [_ai_with_calls(_call("s", {}, "c1"))]}})

    error_message = ToolMessage(content="it broke", tool_call_id="c1", status="error")
    events = norm.handle("updates", {"tools": {"messages": [error_message]}})

    assert types_of(events) == ["tool_call_failed"]
    assert events[0].data == {"error": "it broke", "name": "s"}


def test_an_uncorrelated_tool_result_still_completes():
    """A result whose start was never seen should render, not vanish."""
    norm = new_normalizer()

    events = norm.handle(
        "updates", {"tools": {"messages": [ToolMessage(content="R", tool_call_id="orphan")]}}
    )

    assert types_of(events) == ["tool_call_completed"]
    assert events[0].data == {"result": "R"}


# ---- updates channel: messages and state --------------------------------


def test_final_message_on_updates_completes_a_streamed_message():
    norm = new_normalizer()
    norm.handle("messages", (AIMessageChunk(content="Hel", id="m1"), MESSAGES_META))

    events = norm.handle("updates", {"model": {"messages": [AIMessage(content="Hello", id="m1")]}})

    assert types_of(events) == ["message_completed"]
    assert events[0].data == {"text": "Hello"}


def _cited_message(text: str, citations: list[dict[str, Any]], message_id: str) -> AIMessage:
    """An AIMessage whose content is a text block with real Citation annotations,
    matching langchain_core.messages.content's standard shape (verified)."""
    return AIMessage(
        content=[{"type": "text", "text": text, "annotations": citations}], id=message_id
    )


def _citation(**kwargs: Any) -> dict[str, Any]:
    return {"type": "citation", "id": "c1", **kwargs}


def test_citations_are_extracted_on_message_completed():
    norm = new_normalizer()
    norm.handle("messages", (AIMessageChunk(content="", id="m1"), MESSAGES_META))
    message = _cited_message(
        "See the docs.",
        [_citation(url="https://docs.example.com", title="Docs")],
        "m1",
    )

    events = norm.handle("updates", {"model": {"messages": [message]}})

    assert types_of(events) == ["message_completed"]
    assert events[0].data == {
        "text": "See the docs.",
        "citations": [{"url": "https://docs.example.com", "title": "Docs"}],
    }


def test_a_message_with_no_citations_carries_no_citations_key():
    """Absent, not an empty list — the UI should not render a sources section for
    every plain answer."""
    norm = new_normalizer()
    norm.handle("messages", (AIMessageChunk(content="", id="m1"), MESSAGES_META))

    events = norm.handle("updates", {"model": {"messages": [AIMessage(content="plain", id="m1")]}})

    assert events[0].data == {"text": "plain"}
    assert "citations" not in (events[0].data or {})


def test_multiple_citations_on_one_message_are_all_extracted():
    norm = new_normalizer()
    norm.handle("messages", (AIMessageChunk(content="", id="m1"), MESSAGES_META))
    message = _cited_message(
        "Two sources.",
        [_citation(url="https://a.example.com"), _citation(url="https://b.example.com")],
        "m1",
    )

    events = norm.handle("updates", {"model": {"messages": [message]}})

    assert events[0].data is not None
    urls = [c["url"] for c in events[0].data["citations"]]
    assert urls == ["https://a.example.com", "https://b.example.com"]


def test_a_citation_with_start_and_end_index_is_preserved():
    """Lets a richer UI later highlight the exact cited span in the answer."""
    norm = new_normalizer()
    norm.handle("messages", (AIMessageChunk(content="", id="m1"), MESSAGES_META))
    message = _cited_message(
        "The answer is 42.",
        [_citation(cited_text="42", start_index=14, end_index=16)],
        "m1",
    )

    events = norm.handle("updates", {"model": {"messages": [message]}})

    assert events[0].data is not None
    assert events[0].data["citations"] == [{"cited_text": "42", "start_index": 14, "end_index": 16}]


def test_a_non_citation_annotation_is_ignored_not_stringified():
    """NonStandardAnnotation is a real second annotation type in langchain-core;
    only recognized citation shapes should reach the client."""
    norm = new_normalizer()
    norm.handle("messages", (AIMessageChunk(content="", id="m1"), MESSAGES_META))
    message = AIMessage(
        content=[
            {
                "type": "text",
                "text": "hi",
                "annotations": [{"type": "non_standard_annotation", "value": {"custom": 1}}],
            }
        ],
        id="m1",
    )

    events = norm.handle("updates", {"model": {"messages": [message]}})

    assert "citations" not in (events[0].data or {})


def test_citations_never_appear_on_a_message_delta():
    """Citations are metadata on the final message, not something that streams
    token-by-token — they must only ever show up on message_completed."""
    norm = new_normalizer()

    events = norm.handle("messages", (AIMessageChunk(content="partial", id="m1"), MESSAGES_META))

    assert all("citations" not in (e.data or {}) for e in events)


def test_a_message_never_started_does_not_complete():
    """Non-streaming models emit only on `updates`; completing a message the UI
    never opened would render an orphan bubble."""
    norm = new_normalizer()

    events = norm.handle("updates", {"model": {"messages": [AIMessage(content="hi", id="m9")]}})

    assert events == []


def test_state_only_updates_produce_nothing():
    norm = new_normalizer()

    assert norm.handle("updates", {"model": {"counter": 3}}) == []


def test_a_single_message_not_in_a_list_is_tolerated():
    norm = new_normalizer()

    events = norm.handle("updates", {"model": {"messages": _ai_with_calls(_call("s", {}, "c1"))}})

    assert types_of(events) == ["tool_call_started"]


def test_malformed_updates_payload_is_reported():
    with pytest.raises(EventNormalizationError, match="must yield a \\{node_name: state\\} dict"):
        new_normalizer().handle("updates", ["not", "a", "dict"])


# ---- Interrupts ----------------------------------------------------------


class FakeInterrupt:
    """Mirrors langgraph 1.2.11's Interrupt: exactly `value` and `id`."""

    def __init__(self, value: object, id: str) -> None:  # noqa: A002
        self.value = value
        self.id = id


def test_interrupt_arrives_as_a_tuple_under_a_reserved_key():
    """Verified shape: `{'__interrupt__': (Interrupt(...),)}`."""
    norm = new_normalizer()

    events = norm.handle(
        "updates", {"__interrupt__": (FakeInterrupt({"question": "Approve?"}, "i1"),)}
    )

    assert types_of(events) == ["interrupt_created"]
    assert events[0].data == {"interrupt_id": "i1", "value": {"question": "Approve?"}}
    assert events[0].thread_id == "t1", "resuming requires the thread_id"


def test_multiple_interrupts_each_emit():
    norm = new_normalizer()

    events = norm.handle(
        "updates",
        {"__interrupt__": (FakeInterrupt({"q": 1}, "i1"), FakeInterrupt({"q": 2}, "i2"))},
    )

    assert types_of(events) == ["interrupt_created", "interrupt_created"]


def test_an_interrupt_key_is_not_treated_as_a_node():
    """`__interrupt__` is reserved, not a node name; treating it as one would try
    to read `.messages` off a tuple."""
    norm = new_normalizer()

    events = norm.handle("updates", {"__interrupt__": (FakeInterrupt({"q": 1}, "i1"),)})

    assert all(e.node_name is None for e in events)


# ---- Run lifecycle -------------------------------------------------------


def test_run_completed_closes_a_message_left_open_by_the_stream():
    """A stream can end without a final chunk; otherwise the UI streams forever."""
    norm = new_normalizer()
    norm.handle("messages", (AIMessageChunk(content="partial", id="m1"), MESSAGES_META))

    events = norm.run_completed()

    assert types_of(events) == ["message_completed", "run_completed"]


def test_run_failed_reports_message_and_type_without_a_traceback():
    events = new_normalizer().run_failed(ValueError("boom"))

    assert types_of(events) == ["run_failed"]
    assert events[0].data == {"error": "boom", "error_type": "ValueError"}


def test_run_failed_falls_back_to_the_class_name_for_a_blank_message():
    """An empty error string in the UI reads as 'something failed, unspecified'."""
    events = new_normalizer().run_failed(ValueError())

    assert events[0].data is not None and events[0].data["error"] == "ValueError"


def test_a_propagating_exception_fails_the_open_tool_calls():
    """LangGraph re-raises non-ToolInvocationError tool failures (verified), so
    without this the tool card spins forever."""
    norm = new_normalizer()
    norm.handle(
        "updates",
        {"model": {"messages": [_ai_with_calls(_call("boom", {}, "c1"), _call("boom", {}, "c2"))]}},
    )

    events = norm.tool_call_failed(ValueError("exploded"))

    assert types_of(events) == ["tool_call_failed", "tool_call_failed"]
    assert {e.tool_call_id for e in events} == {"c1", "c2"}


def test_already_completed_calls_are_not_failed_again():
    norm = new_normalizer()
    norm.handle("updates", {"model": {"messages": [_ai_with_calls(_call("s", {}, "c1"))]}})
    norm.handle("updates", {"tools": {"messages": [ToolMessage(content="R", tool_call_id="c1")]}})

    assert norm.tool_call_failed(ValueError("late")) == []


# ---- Unknown channels ----------------------------------------------------


def test_an_unknown_stream_mode_is_ignored():
    """A future LangGraph channel must not break a running app."""
    assert new_normalizer().handle("some_future_mode", {"anything": 1}) == []
