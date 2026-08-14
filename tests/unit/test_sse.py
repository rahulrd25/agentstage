"""SSE encoding. The framing is a wire contract: a bare newline breaks a frame."""

from __future__ import annotations

import json

from agentstage.events import AgentEvent
from agentstage.runtime.sse import SSE_HEADERS, format_comment, format_sse


def test_a_frame_carries_type_id_and_data():
    event = AgentEvent.message_delta("r1", message_id="m1", text="hi", sequence=3)

    frame = format_sse(event)

    assert frame.startswith("event: message_delta\n")
    assert "id: 3\n" in frame
    assert '"text": "hi"' in frame or '"text":"hi"' in frame


def test_a_frame_ends_with_a_blank_line():
    """Frames are separated by a blank line; without it a client never dispatches."""
    frame = format_sse(AgentEvent.run_started("r1"))

    assert frame.endswith("\n\n")


def test_the_data_line_is_parseable_json():
    event = AgentEvent.tool_call_started("r1", tool_call_id="c1", name="search", args={"q": "x"})

    data_line = next(line for line in format_sse(event).splitlines() if line.startswith("data: "))
    payload = json.loads(data_line[len("data: ") :])

    assert payload["type"] == "tool_call_started"
    assert payload["data"] == {"name": "search", "args": {"q": "x"}}


def test_multiline_content_never_breaks_the_framing():
    """A literal newline in the payload would be read as a field separator, so
    JSON escaping is what keeps a multi-line tool result safe on the wire."""
    event = AgentEvent.tool_call_completed("r1", tool_call_id="c1", result="line1\nline2\nline3")

    frame = format_sse(event)
    body = frame[: -len("\n\n")]

    assert len([line for line in body.splitlines() if line.startswith("data: ")]) == 1


def test_reasoning_is_stripped_at_the_transport_boundary():
    """Defense in depth: even if a caller forgets to sanitize, the wire is clean."""
    event = AgentEvent(
        type="message_completed",
        run_id="r1",
        message_id="m1",
        data={"text": "answer", "reasoning": "secret"},
    )

    frame = format_sse(event)

    assert "secret" not in frame
    assert "reasoning" not in frame


def test_an_event_without_a_sequence_omits_the_id_line():
    frame = format_sse(AgentEvent.run_started("r1"))

    assert "id:" not in frame


def test_a_comment_frame_is_a_valid_keepalive():
    """A comment is ignored by the client but stops a proxy closing an idle
    connection during a long model pause."""
    frame = format_comment("keepalive")

    assert frame.startswith(":")
    assert frame.endswith("\n\n")


def test_buffering_is_disabled_in_the_headers():
    """Without X-Accel-Buffering, nginx buffers the whole response and streaming
    silently degrades to one payload at the end."""
    assert SSE_HEADERS["X-Accel-Buffering"] == "no"
    assert "no-cache" in SSE_HEADERS["Cache-Control"]
