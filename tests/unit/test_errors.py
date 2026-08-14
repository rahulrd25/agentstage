"""The exception hierarchy is part of the public contract."""

from __future__ import annotations

import pytest

from agentstage.errors import (
    AdapterError,
    AgentRunError,
    AgentStageError,
    ConfigError,
    EventNormalizationError,
    RunCancelledError,
)

ALL_ERRORS = [
    ConfigError,
    AdapterError,
    EventNormalizationError,
    AgentRunError,
    RunCancelledError,
]


@pytest.mark.parametrize("error_cls", ALL_ERRORS)
def test_every_error_is_catchable_as_the_base(error_cls: type[AgentStageError]):
    """One `except AgentStageError` must trap everything this library raises."""
    with pytest.raises(AgentStageError):
        raise error_cls("boom")


@pytest.mark.parametrize("error_cls", ALL_ERRORS)
def test_message_is_preserved(error_cls: type[AgentStageError]):
    """Requirement 12: errors must carry an actionable message, not swallow it."""
    assert str(error_cls("actionable detail")) == "actionable detail"


def test_agent_run_error_preserves_the_original_cause():
    """The user's agent traceback must survive being wrapped."""
    original = RuntimeError("the agent exploded")

    with pytest.raises(AgentRunError) as caught:
        try:
            raise original
        except RuntimeError as exc:
            raise AgentRunError("agent run failed") from exc

    assert caught.value.__cause__ is original
