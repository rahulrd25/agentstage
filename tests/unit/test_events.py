"""Event model: construction, validation, serialization, safety, debug output."""

from __future__ import annotations

import json
from typing import get_args

import pytest

from agentstage.errors import EventNormalizationError
from agentstage.events import EVENT_TYPES, AgentEvent
from agentstage.types import EventType

# ---- Contract ------------------------------------------------------------


def test_runtime_event_types_match_the_literal():
    """`Literal` is erased at runtime, so EVENT_TYPES mirrors it by hand.

    This is the guard against the two drifting apart.
    """
    assert frozenset(get_args(EventType)) == EVENT_TYPES


def test_field_order_is_stable():
    """Requirement 16. Reordering or renaming breaks every serialized consumer;
    appending to the end is the only backward-compatible change."""
    assert AgentEvent.FIELDS == (
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


# ---- Validation ----------------------------------------------------------


def test_unknown_event_type_is_rejected_with_the_valid_set():
    with pytest.raises(EventNormalizationError, match="Unknown event type"):
        AgentEvent(type="wat", run_id="r1")  # type: ignore[arg-type]


def test_the_error_lists_what_would_have_been_valid():
    """Requirement 12: the message must be actionable, not just a rejection."""
    with pytest.raises(EventNormalizationError) as caught:
        AgentEvent(type="nope", run_id="r1")  # type: ignore[arg-type]
    assert "run_started" in str(caught.value)


def test_empty_run_id_is_rejected():
    with pytest.raises(EventNormalizationError, match="empty run_id"):
        AgentEvent(type="run_started", run_id="")


def test_negative_sequence_is_rejected():
    with pytest.raises(EventNormalizationError, match="non-negative"):
        AgentEvent(type="run_started", run_id="r1", sequence=-1)


def test_sequence_zero_is_allowed():
    """Off-by-one guard: the first event in a run is sequence 0, not 1."""
    assert AgentEvent(type="run_started", run_id="r1", sequence=0).sequence == 0


@pytest.mark.parametrize(
    "event_type",
    ["tool_call_started", "tool_call_delta", "tool_call_completed", "tool_call_failed"],
)
def test_tool_events_require_a_tool_call_id(event_type: EventType):
    """Without it a result cannot be matched to the card that started it."""
    with pytest.raises(EventNormalizationError, match="requires a tool_call_id"):
        AgentEvent(type=event_type, run_id="r1")


@pytest.mark.parametrize("event_type", ["message_started", "message_delta", "message_completed"])
def test_message_events_require_a_message_id(event_type: EventType):
    """Without it every streamed token would render as a separate message."""
    with pytest.raises(EventNormalizationError, match="requires a message_id"):
        AgentEvent(type=event_type, run_id="r1")


def test_non_message_non_tool_events_need_neither_id():
    AgentEvent(type="state_updated", run_id="r1")
    AgentEvent(type="progress_updated", run_id="r1")


def test_events_are_immutable():
    """The transport may hand one event to several consumers; none may mutate it."""
    event = AgentEvent(type="run_started", run_id="r1")
    with pytest.raises((AttributeError, TypeError)):
        event.run_id = "r2"  # type: ignore[misc]


# ---- Serialization -------------------------------------------------------


def test_to_dict_omits_unset_fields():
    """SSE frames are sent per token; absent fields must not cost bytes."""
    payload = AgentEvent(type="run_started", run_id="r1").to_dict()
    assert payload == {"type": "run_started", "run_id": "r1"}


def test_round_trip_preserves_every_field():
    original = AgentEvent(
        type="tool_call_completed",
        run_id="r1",
        thread_id="t1",
        node_name="tools",
        message_id="m1",
        tool_call_id="call_1",
        sequence=7,
        data={"result": "RESULT for x", "name": "search"},
        metadata={"langgraph_step": 2},
    )

    assert AgentEvent.from_dict(original.to_dict()) == original


def test_json_round_trip_preserves_every_field():
    original = AgentEvent.tool_call_started(
        "r1", tool_call_id="call_1", name="search", args={"query": "x"}, sequence=3
    )

    assert AgentEvent.from_json(original.to_json()) == original


def test_unknown_fields_are_rejected_rather_than_ignored():
    """A silently dropped field is a bug that surfaces much later."""
    with pytest.raises(EventNormalizationError, match="Unknown event field"):
        AgentEvent.from_dict({"type": "run_started", "run_id": "r1", "surprise": 1})


def test_missing_type_is_reported_specifically():
    with pytest.raises(EventNormalizationError, match="missing the required 'type'"):
        AgentEvent.from_dict({"run_id": "r1"})


def test_missing_run_id_is_reported_specifically():
    with pytest.raises(EventNormalizationError, match="missing the required 'run_id'"):
        AgentEvent.from_dict({"type": "run_started"})


def test_malformed_json_raises_our_error_not_a_json_error():
    """Callers catch AgentStageError; a raw JSONDecodeError would escape that."""
    with pytest.raises(EventNormalizationError, match="not valid JSON"):
        AgentEvent.from_json("{not json")


def test_json_array_payload_is_rejected():
    with pytest.raises(EventNormalizationError, match="must be a JSON object"):
        AgentEvent.from_json("[1, 2, 3]")


def test_non_serializable_tool_result_degrades_instead_of_killing_the_run():
    """A tool may return any object. Requirement 13 says surface it, but a repr in
    the UI beats a crashed stream."""

    class Opaque:
        def __str__(self) -> str:
            return "opaque-object"

    event = AgentEvent.tool_call_completed("r1", tool_call_id="c1", result=Opaque())

    assert json.loads(event.to_json())["data"]["result"] == "opaque-object"


def test_unicode_survives_serialization():
    event = AgentEvent.message_delta("r1", message_id="m1", text="café 日本語")

    assert AgentEvent.from_json(event.to_json()).data == {"text": "café 日本語"}


# ---- Safety --------------------------------------------------------------


@pytest.mark.parametrize(
    "forbidden",
    [
        "reasoning",
        "reasoning_content",
        "thinking",
        "thought",
        "chain_of_thought",
        "additional_kwargs",
    ],
)
def test_sanitized_strips_chain_of_thought(forbidden: str):
    """Security requirement: hidden reasoning must never reach the browser."""
    event = AgentEvent(
        type="message_completed",
        run_id="r1",
        message_id="m1",
        data={"text": "the answer", forbidden: "secret internal reasoning"},
    )

    clean = event.sanitized

    assert clean.data == {"text": "the answer"}
    assert forbidden not in (clean.to_json())


def test_sanitized_keeps_safe_fields_and_identity():
    event = AgentEvent.tool_call_completed("r1", tool_call_id="c1", result="ok", name="search")

    assert event.sanitized is event, "no forbidden keys means no copy"


def test_sanitized_handles_events_with_no_data():
    event = AgentEvent(type="run_started", run_id="r1")

    assert event.sanitized is event


def test_sanitizing_does_not_mutate_the_original():
    event = AgentEvent(
        type="message_completed",
        run_id="r1",
        message_id="m1",
        data={"text": "hi", "reasoning": "secret"},
    )

    clean = event.sanitized

    assert clean.data == {"text": "hi"}
    assert event.data is not None and "reasoning" in event.data


# ---- Debug output --------------------------------------------------------


def test_describe_is_a_single_scannable_line():
    event = AgentEvent.tool_call_started(
        "run-abcdefgh-1234",
        tool_call_id="call_1",
        name="search",
        args={"query": "x"},
        node_name="tools",
        sequence=4,
    )

    line = event.describe()

    assert "\n" not in line
    assert "#4" in line
    assert "tool_call_started" in line
    assert "node=tools" in line
    assert "call_1" in line


def test_describe_truncates_long_values():
    event = AgentEvent.message_delta("r1", message_id="m1", text="x" * 500)

    line = event.describe()

    assert len(line) < 150
    assert "…" in line


def test_describe_collapses_newlines_in_payloads():
    """A multi-line tool result must not break one-event-per-line log scanning."""
    event = AgentEvent.tool_call_completed("r1", tool_call_id="c1", result="a\nb\nc")

    assert "\n" not in event.describe()


def test_describe_keeps_a_whitespace_only_delta_visible():
    """A streamed token can be a single space. Rendering it as an empty string
    makes a correct stream look broken when reading logs."""
    event = AgentEvent.message_delta("r1", message_id="m1", text=" ")

    assert "' '" in event.describe()


def test_describe_shows_leading_and_trailing_whitespace():
    event = AgentEvent.message_delta("r1", message_id="m1", text="  padded  ")

    assert "'  padded  '" in event.describe()


def test_describe_leaves_ordinary_text_unquoted():
    """Quoting everything would add noise to the common case."""
    event = AgentEvent.message_delta("r1", message_id="m1", text="hello")

    assert "text=hello" in event.describe()


# ---- Constructors --------------------------------------------------------


def test_run_failed_carries_the_message_and_type():
    event = AgentEvent.run_failed("r1", error="boom", error_type="RuntimeError")

    assert event.type == "run_failed"
    assert event.data == {"error": "boom", "error_type": "RuntimeError"}


def test_message_delta_sends_only_the_delta():
    """Performance requirement: never send the accumulated message per token."""
    event = AgentEvent.message_delta("r1", message_id="m1", text="lo")

    assert event.data == {"text": "lo"}


def test_tool_call_started_defaults_args_to_empty_dict():
    """The UI renders args unconditionally; None would force a null check there."""
    event = AgentEvent.tool_call_started("r1", tool_call_id="c1", name="search")

    assert event.data == {"name": "search", "args": {}}


def test_interrupt_created_carries_id_and_value():
    event = AgentEvent.interrupt_created(
        "r1", interrupt_id="i1", value={"question": "Approve?"}, thread_id="t1"
    )

    assert event.data == {"interrupt_id": "i1", "value": {"question": "Approve?"}}
    assert event.thread_id == "t1", "resuming requires the thread_id"


def test_constructors_accept_sequence_and_metadata_passthrough():
    event = AgentEvent.run_started("r1", thread_id="t1", sequence=0, metadata={"origin": "test"})

    assert event.sequence == 0
    assert event.metadata == {"origin": "test"}
