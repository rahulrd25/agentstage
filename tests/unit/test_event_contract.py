"""Golden-file contract test for the wire format.

Requirement 16: once the API is public, the serialized shape is a promise. This
pins the exact JSON for one event of every type. A diff here means a consumer
would break — adding an optional field is fine and won't trip this; renaming,
removing, or reordering will.

If a change here is intentional, update the golden file in the same commit so the
break is visible in review rather than discovered by a client.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentstage.events import EVENT_TYPES, AgentEvent

GOLDEN_PATH = Path(__file__).parent / "golden_events.json"


def _sample_events() -> dict[str, AgentEvent]:
    """One representative event per type, with every field populated."""
    return {
        "run_started": AgentEvent.run_started("r1", thread_id="t1", sequence=0),
        "run_completed": AgentEvent.run_completed("r1", thread_id="t1", sequence=9),
        "run_failed": AgentEvent.run_failed(
            "r1", error="boom", error_type="RuntimeError", thread_id="t1", sequence=9
        ),
        "message_started": AgentEvent(
            type="message_started", run_id="r1", message_id="m1", node_name="model", sequence=1
        ),
        "message_delta": AgentEvent.message_delta(
            "r1", message_id="m1", text="hel", node_name="model", sequence=2
        ),
        "message_completed": AgentEvent(
            type="message_completed",
            run_id="r1",
            message_id="m1",
            node_name="model",
            sequence=3,
            data={"text": "hello"},
        ),
        "tool_call_started": AgentEvent.tool_call_started(
            "r1",
            tool_call_id="call_1",
            name="search",
            args={"query": "x"},
            node_name="model",
            sequence=4,
        ),
        "tool_call_delta": AgentEvent(
            type="tool_call_delta",
            run_id="r1",
            tool_call_id="call_1",
            node_name="model",
            sequence=5,
            data={"args_delta": '{"que'},
        ),
        "tool_call_completed": AgentEvent.tool_call_completed(
            "r1",
            tool_call_id="call_1",
            result="RESULT for x",
            name="search",
            node_name="tools",
            sequence=6,
        ),
        "tool_call_failed": AgentEvent.tool_call_failed(
            "r1",
            tool_call_id="call_1",
            error="tool exploded",
            name="search",
            node_name="tools",
            sequence=6,
        ),
        "interrupt_created": AgentEvent.interrupt_created(
            "r1",
            interrupt_id="i1",
            value={"question": "Approve?"},
            thread_id="t1",
            sequence=7,
        ),
        "progress_updated": AgentEvent(
            type="progress_updated",
            run_id="r1",
            node_name="model",
            sequence=8,
            data={"message": "searching"},
        ),
        "state_updated": AgentEvent(
            type="state_updated",
            run_id="r1",
            thread_id="t1",
            node_name="model",
            sequence=8,
            data={"keys": ["messages"]},
        ),
    }


def test_a_sample_exists_for_every_event_type():
    """Guards the guard: a new event type must be added to the golden file too."""
    assert set(_sample_events()) == set(EVENT_TYPES)


def test_wire_format_matches_the_golden_file():
    expected = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    actual = {name: event.to_dict() for name, event in _sample_events().items()}

    assert actual == expected, (
        "The serialized event format changed. If intentional, regenerate "
        "tests/unit/golden_events.json in this commit; otherwise this is a "
        "backward-compatibility break for every client."
    )


@pytest.mark.parametrize("name", sorted(_sample_events()))
def test_every_event_type_round_trips_through_json(name: str):
    original = _sample_events()[name]

    assert AgentEvent.from_json(original.to_json()) == original
