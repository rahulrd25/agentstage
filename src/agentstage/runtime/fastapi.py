"""FastAPI integration — the HTTP surface for an agent.

Mountable into an existing application so agentstage does not own the user's
server:

    api = build_router(config)
    my_app.include_router(api, prefix="/agent")

Security posture: the agent runs server-side only, API keys never reach the
browser, and client-supplied identity is never trusted. ``thread_id`` arrives from
the client because a conversation must be resumable, but the ``authenticate`` hook
is where an application maps a request to a *user* and scopes threads to them.
Row-level authorization is the application's responsibility — see ``docs``.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from agentstage.adapters.langgraph import LangGraphAdapter
from agentstage.errors import AgentStageError
from agentstage.events.models import AgentEvent
from agentstage.files import AttachmentStore, content_block_for
from agentstage.runtime.sse import SSE_HEADERS, format_comment, format_sse
from agentstage.storage import ThreadNotFoundError, ThreadStore

__all__ = ["AppConfig", "ChatRequest", "build_router"]

#: Client-visible identity for attachment ownership. `str(result)` of whatever
#: `authenticate` returns — agentstage does not know the shape of an app's user
#: object, only that two uploads from the "same" caller should compare equal.
_UNAUTHENTICATED_OWNER = None

#: Emitted while the agent is quiet, so proxies do not close an idle connection.
_KEEPALIVE_SECONDS = 15.0


AuthenticateHook = Callable[[Request], Awaitable[Any] | Any]


@dataclass
class AppConfig:
    """Server-side configuration for a mounted agent.

    ``authenticate`` is a hook, not an implementation. It receives the request and
    may raise to reject it; whatever it returns is available to the application.
    agentstage does not pretend to provide authorization.
    """

    adapter: LangGraphAdapter
    title: str = "Agent"
    authenticate: AuthenticateHook | None = None
    #: Extra config forwarded to every agent run (never client-controlled).
    run_config: dict[str, Any] = field(default_factory=dict)
    #: None disables the /upload and attachment_ids surface entirely — an app
    #: that never calls AgentApp.chat(attachments=True) gets no upload endpoint.
    attachments: AttachmentStore | None = None
    #: None disables the /threads surface entirely — an app that never calls
    #: AgentApp.thread_history(enabled=True) gets no thread-listing endpoint.
    threads: ThreadStore | None = None


class ChatRequest(BaseModel):
    """A chat submission.

    ``thread_id`` is client-supplied so a conversation can resume, and is
    therefore untrusted: scope it to an authenticated user in ``authenticate``
    before treating it as an identity.
    """

    message: str = Field(min_length=1, max_length=32_000)
    thread_id: str | None = Field(default=None, max_length=200)
    #: IDs returned by a prior POST /upload. Capped generously above what a
    #: single message plausibly attaches; the store's own size limit is what
    #: actually bounds the data.
    attachment_ids: list[str] = Field(default_factory=list, max_length=20)


class ResumeRequest(BaseModel):
    """A response to a pending human-in-the-loop interrupt.

    ``decision`` is a plain string rather than a boolean: LangGraph's
    ``interrupt()`` receives whatever value ``Command(resume=...)`` carries, and
    the interrupted node decides what it means. For a hand-rolled ``interrupt()``
    call in a custom graph node that is a plain bool for approve/reject, and the
    raw ``edited_value`` for a replacement payload (verified). A prebuilt
    middleware can impose its own protocol instead — see
    ``_build_resume_value``, which detects that case and builds whatever that
    middleware actually expects rather than assuming one contract fits both.
    """

    thread_id: str = Field(min_length=1, max_length=200)
    decision: str = Field(pattern=r"^(approved|rejected|edited)$")
    edited_value: Any = None


class RenameThreadRequest(BaseModel):
    title: str = Field(min_length=1, max_length=200)


def build_router(config: AppConfig) -> APIRouter:
    """Build the agent's HTTP router."""
    router = APIRouter()

    async def _authenticate(request: Request) -> str | None:
        """Run the hook and return an owner key for attachment scoping.

        The hook's return value is application-defined and may not be a plain
        string (a user object, a dict, ...), so it is stringified here rather than
        stored as-is — attachments only need equality, never the object itself.
        """
        if config.authenticate is None:
            return _UNAUTHENTICATED_OWNER
        result = config.authenticate(request)
        if asyncio.iscoroutine(result):
            result = await result
        return str(result) if result is not None else _UNAUTHENTICATED_OWNER

    @router.get("/health")
    async def health() -> dict[str, Any]:
        """Liveness probe. Deliberately unauthenticated and free of detail.

        ``attachments``/``threads`` tell the UI whether to show the upload
        control and the conversation sidebar — the alternative is always
        showing them and every click 404ing on an app that never enabled the
        feature.
        """
        return {
            "status": "ok",
            "title": config.title,
            "attachments": config.attachments is not None,
            "threads": config.threads is not None,
        }

    @router.post("/upload")
    async def upload(request: Request, file: UploadFile = File(...)) -> dict[str, Any]:
        """Accept a file, returning an id the client references from POST /chat.

        404 if attachments are disabled — not 403 — so an app that never enabled
        the feature doesn't leak that an upload endpoint exists at all.
        """
        if config.attachments is None:
            raise HTTPException(status_code=404, detail="Not found.")

        owner = await _authenticate(request)
        data = await file.read()
        try:
            attachment = await config.attachments.save(
                filename=file.filename or "upload",
                content_type=file.content_type or "application/octet-stream",
                data=data,
                owner=owner,
            )
        except AgentStageError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        return {
            "id": attachment.id,
            "filename": attachment.filename,
            "content_type": attachment.content_type,
            "size": attachment.size,
        }

    @router.post("/chat")
    async def chat(request: Request, body: ChatRequest) -> StreamingResponse:
        """Run the agent and stream normalized events as SSE.

        Returns 200 with an event stream. A failure *during* the run arrives as a
        ``run_failed`` event rather than an HTTP error, because the response has
        already begun — the client learns about it in-band.
        """
        owner = await _authenticate(request)

        if body.attachment_ids and config.attachments is None:
            raise HTTPException(status_code=400, detail="This agent does not accept attachments.")

        content_blocks = []
        for attachment_id in body.attachment_ids:
            assert config.attachments is not None  # guarded above
            try:
                attachment, data = await config.attachments.read(attachment_id, owner=owner)
            except AgentStageError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            content_blocks.append(content_block_for(attachment, data))

        thread_id = (
            require_thread_id(body.thread_id)
            if body.thread_id
            else f"thread-{uuid.uuid4().hex[:12]}"
        )
        run_id = f"run-{uuid.uuid4().hex[:12]}"

        if config.threads is not None:
            try:
                await config.threads.touch(thread_id, owner=owner, title=_title_from(body.message))
            except ThreadNotFoundError as exc:
                # A client-supplied thread_id belonging to a different owner.
                # Same status as a genuinely unknown attachment_id: the failure
                # mode is identical from the caller's point of view.
                raise HTTPException(status_code=404, detail=str(exc)) from exc

        producer = config.adapter.stream(
            body.message,
            thread_id=thread_id,
            run_id=run_id,
            config=config.run_config,
            content_blocks=content_blocks or None,
        )
        return StreamingResponse(
            _event_stream(request, producer, run_id=run_id, thread_id=thread_id),
            media_type="text/event-stream",
            headers=SSE_HEADERS,
        )

    @router.post("/resume")
    async def resume(request: Request, body: ResumeRequest) -> StreamingResponse:
        """Answer a pending human-in-the-loop interrupt and stream the rest of the run.

        400 if the thread has no pending interrupt — resuming one silently starts
        an unrelated run instead of failing (verified), so this is the boundary
        that must catch a stale or already-answered thread_id before that happens.
        """
        owner = await _authenticate(request)
        thread_id = require_thread_id(body.thread_id)

        if config.threads is not None:
            try:
                await config.threads.touch(thread_id, owner=owner, title=None)
            except ThreadNotFoundError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc

        if not await config.adapter.has_pending_interrupt(thread_id):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Thread {thread_id!r} has no pending interrupt to resume. It may "
                    "have already been answered, or never paused."
                ),
            )

        interrupt_value = await config.adapter.pending_interrupt_value(thread_id)
        try:
            resume_value = _build_resume_value(
                interrupt_value, decision=body.decision, edited_value=body.edited_value
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        run_id = f"run-{uuid.uuid4().hex[:12]}"
        producer = config.adapter.resume(
            thread_id=thread_id,
            resume_value=resume_value,
            run_id=run_id,
            config=config.run_config,
        )
        return StreamingResponse(
            _event_stream(request, producer, run_id=run_id, thread_id=thread_id),
            media_type="text/event-stream",
            headers=SSE_HEADERS,
        )

    @router.get("/threads")
    async def list_threads(request: Request) -> list[dict[str, Any]]:
        """List the caller's conversations, most recently used first.

        404 if thread history is disabled — not an empty list — so an app that
        never enabled the feature doesn't leak that the endpoint exists.
        """
        if config.threads is None:
            raise HTTPException(status_code=404, detail="Not found.")
        owner = await _authenticate(request)
        threads = await config.threads.list_for_owner(owner)
        return [_serialize_thread(t) for t in threads]

    @router.get("/threads/{thread_id}/messages")
    async def thread_messages(request: Request, thread_id: str) -> list[dict[str, Any]]:
        """Reconstruct a thread's transcript for display when it is opened.

        Ownership is checked against the thread store, not the checkpointer —
        LangGraph's checkpointer has no concept of ownership on its own, so
        without this check any caller who knew or guessed a thread_id could read
        another user's conversation.
        """
        if config.threads is None:
            raise HTTPException(status_code=404, detail="Not found.")
        thread_id = require_thread_id(thread_id)
        owner = await _authenticate(request)
        try:
            await config.threads.get(thread_id, owner=owner)
        except ThreadNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return await config.adapter.get_history(thread_id)

    @router.patch("/threads/{thread_id}")
    async def rename_thread(
        request: Request, thread_id: str, body: RenameThreadRequest
    ) -> dict[str, Any]:
        if config.threads is None:
            raise HTTPException(status_code=404, detail="Not found.")
        thread_id = require_thread_id(thread_id)
        owner = await _authenticate(request)
        try:
            info = await config.threads.rename(thread_id, owner=owner, title=body.title)
        except ThreadNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return _serialize_thread(info)

    @router.delete("/threads/{thread_id}", status_code=204)
    async def delete_thread(request: Request, thread_id: str) -> None:
        """Remove a thread's listing.

        Does not touch the agent's checkpointer — deleting the underlying
        LangGraph state is a separate, more destructive operation this endpoint
        deliberately does not perform.
        """
        if config.threads is None:
            raise HTTPException(status_code=404, detail="Not found.")
        thread_id = require_thread_id(thread_id)
        owner = await _authenticate(request)
        try:
            await config.threads.delete(thread_id, owner=owner)
        except ThreadNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    return router


def _is_hitl_middleware_request(value: Any) -> bool:
    """Whether an interrupt's value is a ``langchain`` ``HumanInTheLoopMiddleware``
    request rather than a hand-rolled ``interrupt(...)`` call's own payload.

    That middleware always resumes through ``interrupt(hitl_request)["decisions"]``
    — a structured ``{"decisions": [...]}`` reply — which a bare bool cannot
    satisfy (``TypeError: 'bool' object is not subscriptable``). Detecting the
    shape it always sends (``action_requests`` + ``review_configs``) is what lets
    ``_build_resume_value`` build the right payload without knowing in advance
    which of the two protocols created a given interrupt.
    """
    return (
        isinstance(value, dict)
        and isinstance(value.get("action_requests"), list)
        and isinstance(value.get("review_configs"), list)
    )


def _build_resume_value(interrupt_value: Any, *, decision: str, edited_value: Any) -> Any:
    """Translate a UI decision into whatever the pending interrupt actually expects.

    A hand-rolled ``interrupt()`` call in a custom graph node (see
    ``examples/human_approval``) can expect anything; agentstage has always
    passed a plain bool for approve/reject and the raw ``edited_value`` through
    unchanged for edit, and that contract is unconditionally preserved here.
    ``HumanInTheLoopMiddleware`` requests are detected and answered in the
    ``{"decisions": [...]}`` shape it actually reads on resume instead.
    """
    if not _is_hitl_middleware_request(interrupt_value):
        return edited_value if decision == "edited" else decision == "approved"

    if decision == "approved":
        item: dict[str, Any] = {"type": "approve"}
    elif decision == "rejected":
        item = {"type": "reject"}
    else:
        if (
            not isinstance(edited_value, dict)
            or "name" not in edited_value
            or "args" not in edited_value
        ):
            msg = (
                "This interrupt was created by HumanInTheLoopMiddleware, which requires "
                "edited_value shaped as {'name': <tool name>, 'args': {...}} to build its "
                f"edited_action, got: {edited_value!r}."
            )
            raise ValueError(msg)
        item = {
            "type": "edit",
            "edited_action": {"name": edited_value["name"], "args": edited_value["args"]},
        }

    # interrupt() can bundle several tool calls needing approval into one call;
    # agentstage's UI currently offers a single decision per interrupt, so that
    # one decision is applied to every action request in the batch.
    return {"decisions": [item for _ in interrupt_value["action_requests"]]}


def _serialize_thread(info: Any) -> dict[str, Any]:
    return {
        "thread_id": info.thread_id,
        "title": info.title,
        "created_at": info.created_at,
        "updated_at": info.updated_at,
    }


def _title_from(message: str) -> str:
    """Derive a default thread title from the first message.

    Only applied when a thread is created — ``ThreadStore.touch`` preserves an
    existing title on every later call, so this never overwrites one a user
    (or a future rename endpoint) already set.
    """
    collapsed = " ".join(message.split())
    return collapsed[:60] + ("…" if len(collapsed) > 60 else "")


async def _event_stream(
    request: Request,
    events: AsyncIterator[AgentEvent],
    *,
    run_id: str,
    thread_id: str,
) -> AsyncIterator[str]:
    """Yield SSE frames for one run.

    Cancellation is cooperative: if the client disconnects, the loop stops pulling
    from the agent and the generator returns, which closes the underlying
    LangGraph stream. That is the cancellation the transport can honor without the
    agent's cooperation.
    """
    queue: asyncio.Queue[AgentEvent | None] = asyncio.Queue()

    async def produce() -> None:
        try:
            async for event in events:
                await queue.put(event)
        except AgentStageError as exc:
            # The adapter converts agent failures into run_failed events, so
            # reaching here means agentstage itself failed. Surface it in-band.
            await queue.put(AgentEvent.run_failed(run_id, error=str(exc), thread_id=thread_id))
        finally:
            await queue.put(None)

    producer = asyncio.create_task(produce())
    try:
        while True:
            if await request.is_disconnected():
                break
            try:
                event = await asyncio.wait_for(queue.get(), timeout=_KEEPALIVE_SECONDS)
            except TimeoutError:
                yield format_comment("keepalive")
                continue
            if event is None:
                break
            yield format_sse(event)
    finally:
        producer.cancel()
        # Await the cancellation so the agent's stream is actually torn down before
        # the response ends, rather than leaking a task. Anything raised during
        # teardown is discarded: the response is already over, so there is nowhere
        # left to report it.
        with suppress(BaseException):
            await producer


def require_thread_id(thread_id: str | None) -> str:
    """Validate a thread id supplied by a client.

    Rejects path separators and control characters so a thread id can never be
    used to traverse a storage path once persistence lands.
    """
    if not thread_id:
        raise HTTPException(status_code=400, detail="thread_id is required.")
    if any(ch in thread_id for ch in "/\\\x00") or not thread_id.isprintable():
        raise HTTPException(status_code=400, detail="thread_id contains invalid characters.")
    return thread_id
