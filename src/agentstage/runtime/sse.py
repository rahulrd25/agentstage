"""Server-Sent Events encoding.

SSE rather than WebSocket: agent streaming is one-way server→client, and SSE gives
auto-reconnect, plain HTTP, and debuggability with ``curl``. Client→server actions
(submit, approve, cancel) are ordinary POSTs.

The wire format is deliberately minimal — one JSON ``AgentEvent`` per frame, with
the event type in the SSE ``event:`` field so a browser can attach per-type
listeners without parsing first.
"""

from __future__ import annotations

from agentstage.events.models import AgentEvent

__all__ = ["SSE_HEADERS", "format_comment", "format_sse"]


#: Headers required for SSE to survive real deployments.
SSE_HEADERS: dict[str, str] = {
    # Without this, nginx and friends buffer the whole response and streaming
    # silently degrades to one big payload at the end.
    "X-Accel-Buffering": "no",
    "Cache-Control": "no-cache, no-transform",
    "Connection": "keep-alive",
}


def format_sse(event: AgentEvent) -> str:
    """Encode one event as an SSE frame.

    The payload is sanitized here — the transport boundary is the last point
    before data reaches the browser, so reasoning cannot leak even if a caller
    forgot.

    ``id:`` carries the sequence number so a reconnecting client can tell the
    server where it left off via ``Last-Event-ID``.
    """
    safe = event.sanitized
    lines = [f"event: {safe.type}"]
    if safe.sequence is not None:
        lines.append(f"id: {safe.sequence}")
    # A payload must never contain a bare newline: SSE treats it as a field break.
    # AgentEvent.to_json() emits compact JSON with escaped newlines, so this holds.
    lines.append(f"data: {safe.to_json()}")
    return "\n".join(lines) + "\n\n"


def format_comment(text: str = "") -> str:
    """Encode an SSE comment, used as a keep-alive.

    Proxies commonly close idle connections; a comment is ignored by the client
    but keeps the socket warm during a long model pause.
    """
    return f": {text}\n\n"
