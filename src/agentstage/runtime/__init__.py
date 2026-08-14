"""Runtime: the HTTP transport that carries events to the browser."""

from agentstage.runtime.sse import SSE_HEADERS, format_comment, format_sse

__all__ = ["SSE_HEADERS", "format_comment", "format_sse"]
