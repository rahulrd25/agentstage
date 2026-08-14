"""Type-contract tests.

These guard the values verified by introspection in Milestone 0. They fail loudly
if a dependency upgrade changes the ground truth the adapter is built on.
"""

from __future__ import annotations

from typing import get_args

from agentstage.types import EventType, StreamMode, SupportsAgentStream


def test_event_types_match_the_published_contract():
    """Requirement 16: this set is the public event contract. Additions are
    backward-compatible; removals and renames are not."""
    assert set(get_args(EventType)) == {
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


def test_stream_modes_match_installed_langgraph():
    """Verified against langgraph 1.2.11 by introspection, not documentation.

    If this fails after an upgrade, re-verify before widening the literal — the
    adapter's channel handling depends on this exact set.
    """
    from langgraph.types import StreamMode as LangGraphStreamMode

    assert set(get_args(StreamMode)) == set(get_args(LangGraphStreamMode))


def test_interrupt_has_exactly_value_and_id():
    """Tutorials reference `resumable`/`ns`/`when`; those do not exist in 1.2.11.

    Pinned here so code is never written against the phantom fields.
    """
    from langgraph.types import Interrupt

    assert set(Interrupt.__dataclass_fields__) == {"value", "id"}


def test_supports_agent_stream_accepts_any_object_with_astream_and_aget_state():
    """Structural typing keeps fakes and real compiled graphs on one code path."""

    class Streamer:
        def astream(self, input, config=None, **kwargs):  # noqa: A002
            raise NotImplementedError

        def aget_state(self, *args, **kwargs):
            raise NotImplementedError

    assert isinstance(Streamer(), SupportsAgentStream)


def test_supports_agent_stream_rejects_astream_without_aget_state():
    """`has_pending_interrupt` needs aget_state; a stub missing it must fail the
    check at config time rather than crashing on first use."""

    class OnlyStreams:
        def astream(self, input, config=None, **kwargs):  # noqa: A002
            raise NotImplementedError

    assert not isinstance(OnlyStreams(), SupportsAgentStream)


def test_supports_agent_stream_rejects_an_uncompiled_object():
    """An object without `astream` must fail the check, so config-time detection
    can raise AdapterError instead of an AttributeError mid-stream."""

    class NotAnAgent:
        def invoke(self, input):  # noqa: A002
            raise NotImplementedError

    assert not isinstance(NotAnAgent(), SupportsAgentStream)
