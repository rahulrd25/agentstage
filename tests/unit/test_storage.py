"""ThreadStore: both implementations must satisfy the same contract, so this
suite runs against each via a fixture rather than duplicating every test."""

from __future__ import annotations

from pathlib import Path

import pytest

from agentstage.storage import (
    InMemoryThreadStore,
    SQLiteThreadStore,
    ThreadNotFoundError,
    ThreadStore,
)


@pytest.fixture(params=["memory", "sqlite"])
def store(request: pytest.FixtureRequest, tmp_path: Path) -> ThreadStore:
    if request.param == "memory":
        return InMemoryThreadStore()
    return SQLiteThreadStore(tmp_path / "threads.sqlite3")


async def test_touching_a_new_thread_creates_it(store: ThreadStore):
    info = await store.touch("t1", owner="u1", title="My chat")

    assert info.thread_id == "t1"
    assert info.title == "My chat"
    assert info.owner == "u1"


async def test_a_thread_with_no_title_gets_a_default():
    """The chat endpoint calls touch() on every run; most runs never pass a
    title, so a thread must still be listable without one."""
    store = InMemoryThreadStore()

    info = await store.touch("t1", owner=None, title=None)

    assert info.title
    assert "t1"[-8:] in info.title


async def test_touching_an_existing_thread_does_not_rename_it(store: ThreadStore):
    """A later run's title (often derived from the latest message) must not
    silently overwrite a title the user already set."""
    await store.touch("t1", owner="u1", title="Original")

    info = await store.touch("t1", owner="u1", title="Different title from a later call")

    assert info.title == "Original"


async def test_touching_updates_updated_at_but_not_created_at(store: ThreadStore):
    first = await store.touch("t1", owner="u1", title="x")
    second = await store.touch("t1", owner="u1", title="x")

    assert second.created_at == first.created_at
    assert second.updated_at >= first.updated_at


async def test_get_returns_the_threads_metadata(store: ThreadStore):
    await store.touch("t1", owner="u1", title="My chat")

    info = await store.get("t1", owner="u1")

    assert info.title == "My chat"


async def test_get_someone_elses_thread_is_rejected(store: ThreadStore):
    """The property a transcript endpoint relies on: get() is the ownership
    check performed before reading message content from the checkpointer, which
    has no concept of ownership on its own."""
    await store.touch("t1", owner="alice", title="secret")

    with pytest.raises(ThreadNotFoundError):
        await store.get("t1", owner="bob")


async def test_get_an_unknown_thread_is_rejected(store: ThreadStore):
    with pytest.raises(ThreadNotFoundError):
        await store.get("no-such-thread", owner="u1")


async def test_listing_returns_only_the_given_owners_threads(store: ThreadStore):
    await store.touch("t1", owner="alice", title="a")
    await store.touch("t2", owner="bob", title="b")

    alice_threads = await store.list_for_owner("alice")

    assert [t.thread_id for t in alice_threads] == ["t1"]


async def test_listing_orders_most_recently_used_first(store: ThreadStore):
    await store.touch("t1", owner="u1", title="first")
    await store.touch("t2", owner="u1", title="second")
    await store.touch("t1", owner="u1", title="first")  # bump t1's updated_at

    threads = await store.list_for_owner("u1")

    assert [t.thread_id for t in threads] == ["t1", "t2"]


async def test_listing_respects_the_limit(store: ThreadStore):
    for i in range(5):
        await store.touch(f"t{i}", owner="u1", title=f"t{i}")

    threads = await store.list_for_owner("u1", limit=2)

    assert len(threads) == 2


async def test_an_unauthenticated_app_uses_none_as_the_owner(store: ThreadStore):
    """An app that never sets `authenticate` has owner=None everywhere; listing
    must still work, not return nothing forever."""
    await store.touch("t1", owner=None, title="x")

    threads = await store.list_for_owner(None)

    assert [t.thread_id for t in threads] == ["t1"]


async def test_touching_someone_elses_thread_id_is_rejected(store: ThreadStore):
    """The load-bearing security property: without this, a second user touching
    a guessed or intercepted thread_id could take over its listing."""
    await store.touch("t1", owner="alice", title="secret")

    with pytest.raises(ThreadNotFoundError):
        await store.touch("t1", owner="bob", title="anything")


async def test_renaming_a_thread_changes_its_title(store: ThreadStore):
    await store.touch("t1", owner="u1", title="old")

    renamed = await store.rename("t1", owner="u1", title="new")

    assert renamed.title == "new"
    listed = await store.list_for_owner("u1")
    assert listed[0].title == "new"


async def test_renaming_someone_elses_thread_is_rejected(store: ThreadStore):
    await store.touch("t1", owner="alice", title="x")

    with pytest.raises(ThreadNotFoundError):
        await store.rename("t1", owner="bob", title="stolen")


async def test_renaming_an_unknown_thread_is_rejected(store: ThreadStore):
    with pytest.raises(ThreadNotFoundError):
        await store.rename("no-such-thread", owner="u1", title="x")


async def test_deleting_a_thread_removes_it_from_listings(store: ThreadStore):
    await store.touch("t1", owner="u1", title="x")

    await store.delete("t1", owner="u1")

    assert await store.list_for_owner("u1") == []


async def test_deleting_someone_elses_thread_is_rejected(store: ThreadStore):
    await store.touch("t1", owner="alice", title="x")

    with pytest.raises(ThreadNotFoundError):
        await store.delete("t1", owner="bob")


async def test_deleting_an_unknown_thread_is_rejected(store: ThreadStore):
    with pytest.raises(ThreadNotFoundError):
        await store.delete("no-such-thread", owner="u1")


# ---- SQLite-specific: durability across instances -------------------------


async def test_sqlite_store_survives_a_new_instance_pointed_at_the_same_file(tmp_path: Path):
    """The whole point of choosing SQLite over the in-memory store: a fresh
    AgentApp built after a process restart must see the same threads."""
    db_path = tmp_path / "threads.sqlite3"
    first = SQLiteThreadStore(db_path)
    await first.touch("t1", owner="u1", title="persisted")

    second = SQLiteThreadStore(db_path)
    threads = await second.list_for_owner("u1")

    assert [t.title for t in threads] == ["persisted"]


async def test_sqlite_store_creates_its_parent_file_on_first_use(tmp_path: Path):
    db_path = tmp_path / "nested" / "threads.sqlite3"
    db_path.parent.mkdir()
    store = SQLiteThreadStore(db_path)

    await store.touch("t1", owner=None, title="x")

    assert db_path.is_file()
