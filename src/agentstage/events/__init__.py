"""Normalized event model — the stable contract between adapters and the UI."""

from agentstage.events.models import EVENT_TYPES, AgentEvent
from agentstage.events.normalize import StreamNormalizer

__all__ = ["EVENT_TYPES", "AgentEvent", "StreamNormalizer"]
