"""Thread metadata storage — what makes conversations listable in the UI.

The distinction that shapes this module: LangGraph's checkpointer is already the
source of truth for *message content* per thread — that's what makes resume work,
and agentstage does not duplicate it. What the checkpointer has no concept of is
metadata a UI needs to list conversations: a title, when a thread was last used,
and who it belongs to. :class:`ThreadStore` is that index, kept deliberately thin.

Session state (the in-flight run, SSE queues) is never persisted here — only a
pointer (``thread_id``) plus display metadata. This is the "clear separation
between session state and persistent state" from the architecture proposal.
"""

from __future__ import annotations

import asyncio
import sqlite3
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

__all__ = ["InMemoryThreadStore", "SQLiteThreadStore", "ThreadInfo", "ThreadStore"]


@dataclass(frozen=True, slots=True)
class ThreadInfo:
    """One conversation's listing metadata. Message content is not here — fetch
    it from the checkpointer via ``thread_id`` when a thread is opened."""

    thread_id: str
    title: str
    owner: str | None
    created_at: float
    updated_at: float


@runtime_checkable
class ThreadStore(Protocol):
    """Where thread listings live. A protocol so a production deployment can
    implement it against its own database without importing agentstage internals.
    """

    async def touch(self, thread_id: str, *, owner: str | None, title: str | None) -> ThreadInfo:
        """Record activity on a thread, creating it if unseen.

        Called once per run so ``updated_at`` and the listing order reflect real
        usage. ``title`` is only applied when the thread is first created — a
        later call with a different title must not silently rename it.
        """
        ...

    async def list_for_owner(self, owner: str | None, *, limit: int = 50) -> Sequence[ThreadInfo]:
        """List an owner's threads, most recently used first."""
        ...

    async def get(self, thread_id: str, *, owner: str | None) -> ThreadInfo:
        """Fetch one thread's metadata. Raises if it does not exist or belongs to
        another owner — this is what a transcript endpoint uses to confirm the
        caller may read the thread before reading its message content from the
        checkpointer, which has no concept of ownership on its own."""
        ...

    async def rename(self, thread_id: str, *, owner: str | None, title: str) -> ThreadInfo:
        """Rename a thread. Raises if it does not exist or belongs to another owner."""
        ...

    async def delete(self, thread_id: str, *, owner: str | None) -> None:
        """Delete a thread's listing. Does not touch the checkpointer — deleting the
        underlying LangGraph state is a separate, more destructive operation the
        application must opt into explicitly."""
        ...


class ThreadNotFoundError(Exception):
    """Referenced a thread that does not exist or isn't owned by the caller.

    Not an ``AgentStageError`` subclass deliberately: this is a storage-layer
    concern, raised identically for "does not exist" and "wrong owner" — same
    reasoning as :class:`agentstage.files.AttachmentNotFoundError` — confirming
    existence to the wrong owner is itself a leak.
    """


def _default_title(thread_id: str) -> str:
    return f"Conversation {thread_id[-8:]}"


@dataclass
class InMemoryThreadStore:
    """The default store: process memory, no persistence across restarts.

    Adequate for local development and the examples. A production deployment
    needs a store backed by real storage — :class:`SQLiteThreadStore` for a
    single-process deployment, or a custom :class:`ThreadStore` for anything
    multi-worker.
    """

    _threads: dict[str, ThreadInfo] = field(default_factory=dict)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def touch(self, thread_id: str, *, owner: str | None, title: str | None) -> ThreadInfo:
        async with self._lock:
            now = _now()
            existing = self._threads.get(thread_id)
            if existing is not None and existing.owner != owner:
                msg = f"Thread {thread_id!r} was not found."
                raise ThreadNotFoundError(msg)
            info = ThreadInfo(
                thread_id=thread_id,
                title=existing.title if existing else (title or _default_title(thread_id)),
                owner=owner,
                created_at=existing.created_at if existing else now,
                updated_at=now,
            )
            self._threads[thread_id] = info
            return info

    async def list_for_owner(self, owner: str | None, *, limit: int = 50) -> Sequence[ThreadInfo]:
        async with self._lock:
            matching = [info for info in self._threads.values() if info.owner == owner]
            matching.sort(key=lambda info: info.updated_at, reverse=True)
            return matching[:limit]

    async def get(self, thread_id: str, *, owner: str | None) -> ThreadInfo:
        async with self._lock:
            info = self._threads.get(thread_id)
            if info is None or info.owner != owner:
                msg = f"Thread {thread_id!r} was not found."
                raise ThreadNotFoundError(msg)
            return info

    async def rename(self, thread_id: str, *, owner: str | None, title: str) -> ThreadInfo:
        async with self._lock:
            info = self._threads.get(thread_id)
            if info is None or info.owner != owner:
                msg = f"Thread {thread_id!r} was not found."
                raise ThreadNotFoundError(msg)
            renamed = ThreadInfo(
                thread_id=info.thread_id,
                title=title,
                owner=info.owner,
                created_at=info.created_at,
                updated_at=info.updated_at,
            )
            self._threads[thread_id] = renamed
            return renamed

    async def delete(self, thread_id: str, *, owner: str | None) -> None:
        async with self._lock:
            info = self._threads.get(thread_id)
            if info is None or info.owner != owner:
                msg = f"Thread {thread_id!r} was not found."
                raise ThreadNotFoundError(msg)
            del self._threads[thread_id]


class SQLiteThreadStore:
    """A durable store backed by stdlib ``sqlite3``.

    ``sqlite3`` is synchronous; every call runs through ``asyncio.to_thread`` so a
    slow disk cannot stall the event loop that is also serving every other
    request. This trades a thread-pool hop per call for zero new dependencies —
    reasonable for the moderate call volume a thread-listing endpoint sees, and
    the same trade-off `anyio`/Starlette make for their own file operations.

    One connection per call, not held open across the instance: an
    :class:`AgentApp` outlives any single request, and a long-held connection is
    exactly what causes "database is locked" errors under concurrent access.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        self._initialized = False
        self._init_lock = asyncio.Lock()

    async def _ensure_schema(self) -> None:
        if self._initialized:
            return
        async with self._init_lock:
            if self._initialized:
                return
            await asyncio.to_thread(self._create_table)
            self._initialized = True

    def _create_table(self) -> None:
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS threads (
                    thread_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    owner TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_threads_owner ON threads(owner)")

    async def touch(self, thread_id: str, *, owner: str | None, title: str | None) -> ThreadInfo:
        await self._ensure_schema()
        return await asyncio.to_thread(self._touch_sync, thread_id, owner, title)

    def _touch_sync(self, thread_id: str, owner: str | None, title: str | None) -> ThreadInfo:
        now = _now()
        with sqlite3.connect(self.path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT * FROM threads WHERE thread_id = ?", (thread_id,)).fetchone()
            if row is not None and row["owner"] != owner:
                msg = f"Thread {thread_id!r} was not found."
                raise ThreadNotFoundError(msg)

            if row is None:
                created_at = now
                resolved_title = title or _default_title(thread_id)
                conn.execute(
                    "INSERT INTO threads (thread_id, title, owner, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (thread_id, resolved_title, owner, created_at, now),
                )
            else:
                created_at = row["created_at"]
                resolved_title = row["title"]
                conn.execute(
                    "UPDATE threads SET updated_at = ? WHERE thread_id = ?", (now, thread_id)
                )
            conn.commit()
            return ThreadInfo(
                thread_id=thread_id,
                title=resolved_title,
                owner=owner,
                created_at=created_at,
                updated_at=now,
            )

    async def list_for_owner(self, owner: str | None, *, limit: int = 50) -> Sequence[ThreadInfo]:
        await self._ensure_schema()
        return await asyncio.to_thread(self._list_sync, owner, limit)

    def _list_sync(self, owner: str | None, limit: int) -> list[ThreadInfo]:
        with sqlite3.connect(self.path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM threads WHERE owner IS ? ORDER BY updated_at DESC LIMIT ?",
                (owner, limit),
            ).fetchall()
            return [
                ThreadInfo(
                    thread_id=row["thread_id"],
                    title=row["title"],
                    owner=row["owner"],
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                )
                for row in rows
            ]

    async def get(self, thread_id: str, *, owner: str | None) -> ThreadInfo:
        await self._ensure_schema()
        return await asyncio.to_thread(self._get_sync, thread_id, owner)

    def _get_sync(self, thread_id: str, owner: str | None) -> ThreadInfo:
        with sqlite3.connect(self.path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT * FROM threads WHERE thread_id = ?", (thread_id,)).fetchone()
            if row is None or row["owner"] != owner:
                msg = f"Thread {thread_id!r} was not found."
                raise ThreadNotFoundError(msg)
            return ThreadInfo(
                thread_id=row["thread_id"],
                title=row["title"],
                owner=row["owner"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
            )

    async def rename(self, thread_id: str, *, owner: str | None, title: str) -> ThreadInfo:
        await self._ensure_schema()
        return await asyncio.to_thread(self._rename_sync, thread_id, owner, title)

    def _rename_sync(self, thread_id: str, owner: str | None, title: str) -> ThreadInfo:
        with sqlite3.connect(self.path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT * FROM threads WHERE thread_id = ?", (thread_id,)).fetchone()
            if row is None or row["owner"] != owner:
                msg = f"Thread {thread_id!r} was not found."
                raise ThreadNotFoundError(msg)
            conn.execute("UPDATE threads SET title = ? WHERE thread_id = ?", (title, thread_id))
            conn.commit()
            return ThreadInfo(
                thread_id=thread_id,
                title=title,
                owner=row["owner"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
            )

    async def delete(self, thread_id: str, *, owner: str | None) -> None:
        await self._ensure_schema()
        await asyncio.to_thread(self._delete_sync, thread_id, owner)

    def _delete_sync(self, thread_id: str, owner: str | None) -> None:
        with sqlite3.connect(self.path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT * FROM threads WHERE thread_id = ?", (thread_id,)).fetchone()
            if row is None or row["owner"] != owner:
                msg = f"Thread {thread_id!r} was not found."
                raise ThreadNotFoundError(msg)
            conn.execute("DELETE FROM threads WHERE thread_id = ?", (thread_id,))
            conn.commit()


def _now() -> float:
    """Wall-clock seconds, used only for ``created_at``/``updated_at`` ordering.

    Not ``time.monotonic()`` (unlike ``files.py``'s TTL clock): these timestamps
    are meant to survive a process restart when using ``SQLiteThreadStore``, and
    monotonic time is only comparable within one process's lifetime.
    """
    return time.time()
