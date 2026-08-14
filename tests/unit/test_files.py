"""In-memory attachment store: size limits, MIME allowlist, magic-byte
verification, owner isolation, and TTL eviction."""

from __future__ import annotations

import pytest

from agentstage.files import (
    Attachment,
    AttachmentNotFoundError,
    AttachmentRejectedError,
    InMemoryAttachmentStore,
    content_block_for,
)

PNG_HEADER = b"\x89PNG\r\n\x1a\n" + b"\x00" * 20
PDF_HEADER = b"%PDF-1.4\n" + b"\x00" * 20


async def test_a_saved_attachment_can_be_read_back():
    store = InMemoryAttachmentStore()

    saved = await store.save(
        filename="notes.txt", content_type="text/plain", data=b"hello", owner="u1"
    )
    attachment, data = await store.read(saved.id, owner="u1")

    assert data == b"hello"
    assert attachment.filename == "notes.txt"
    assert attachment.size == 5


async def test_an_oversized_file_is_rejected():
    store = InMemoryAttachmentStore(max_bytes=10)

    with pytest.raises(AttachmentRejectedError, match="exceeding"):
        await store.save(filename="big.txt", content_type="text/plain", data=b"x" * 11, owner=None)


async def test_a_disallowed_content_type_is_rejected():
    store = InMemoryAttachmentStore()

    with pytest.raises(AttachmentRejectedError, match="not allowed"):
        await store.save(
            filename="app.exe",
            content_type="application/x-msdownload",
            data=b"MZ",
            owner=None,
        )


async def test_content_that_does_not_match_its_declared_type_is_rejected():
    """The Content-Type header is client-supplied. A renamed executable claiming
    to be a PNG must be caught by its actual bytes, not trusted at face value."""
    store = InMemoryAttachmentStore()

    with pytest.raises(AttachmentRejectedError, match="does not match its declared type"):
        await store.save(
            filename="fake.png",
            content_type="image/png",
            data=b"MZ\x90\x00" + b"\x00" * 20,
            owner=None,
        )


@pytest.mark.parametrize(
    ("content_type", "data"),
    [
        ("image/png", PNG_HEADER),
        ("application/pdf", PDF_HEADER),
    ],
)
async def test_content_matching_its_declared_type_is_accepted(content_type: str, data: bytes):
    store = InMemoryAttachmentStore()

    saved = await store.save(filename="f", content_type=content_type, data=data, owner=None)

    assert saved.content_type == content_type


async def test_plaintext_types_have_no_magic_byte_requirement():
    """text/*, application/json have no reliable signature; the MIME allowlist
    and size cap are the only gates for them."""
    store = InMemoryAttachmentStore()

    saved = await store.save(
        filename="data.json", content_type="application/json", data=b'{"a": 1}', owner=None
    )

    assert saved.content_type == "application/json"


async def test_reading_an_unknown_id_is_rejected():
    store = InMemoryAttachmentStore()

    with pytest.raises(AttachmentNotFoundError, match="not found"):
        await store.read("file-does-not-exist", owner=None)


async def test_a_different_owner_cannot_read_the_attachment():
    """Confirming existence to the wrong owner is itself an information leak, so
    this must fail exactly like an unknown id, not a distinct 'forbidden' error."""
    store = InMemoryAttachmentStore()
    saved = await store.save(filename="f", content_type="text/plain", data=b"x", owner="u1")

    with pytest.raises(AttachmentNotFoundError, match="not found"):
        await store.read(saved.id, owner="u2")


async def test_no_owner_can_still_read_an_unowned_attachment():
    """An app that never sets `authenticate` has owner=None everywhere; that must
    keep working, not lock every attachment out."""
    store = InMemoryAttachmentStore()
    saved = await store.save(filename="f", content_type="text/plain", data=b"x", owner=None)

    _, data = await store.read(saved.id, owner=None)

    assert data == b"x"


async def test_an_expired_attachment_cannot_be_read():
    store = InMemoryAttachmentStore(ttl_seconds=0)
    saved = await store.save(filename="f", content_type="text/plain", data=b"x", owner=None)

    with pytest.raises(AttachmentNotFoundError):
        await store.read(saved.id, owner=None)


def test_a_path_traversal_filename_is_reduced_to_its_basename():
    from agentstage.files import _safe_filename

    assert _safe_filename("../../etc/passwd") == "passwd"
    assert _safe_filename("C:\\Users\\evil\\secret.txt") == "secret.txt"


def test_a_filename_with_control_characters_is_sanitized():
    from agentstage.files import _safe_filename

    assert "\x00" not in _safe_filename("bad\x00name.txt")


def test_an_empty_filename_falls_back_to_a_placeholder():
    from agentstage.files import _safe_filename

    assert _safe_filename("") == "upload"


# ---- content_block_for ----------------------------------------------------


def _attachment(content_type: str, filename: str = "f") -> Attachment:
    return Attachment(
        id="file-1", filename=filename, content_type=content_type, size=3, created_at=0.0
    )


def test_an_image_becomes_an_image_content_block():
    block = content_block_for(_attachment("image/png"), b"abc")

    assert block["type"] == "image"
    assert block["mime_type"] == "image/png"
    import base64

    assert base64.b64decode(block["base64"]) == b"abc"


def test_a_pdf_becomes_a_file_content_block():
    block = content_block_for(_attachment("application/pdf", "report.pdf"), b"abc")

    assert block["type"] == "file"
    assert block["mime_type"] == "application/pdf"
    assert block["extras"]["filename"] == "report.pdf"


def test_content_blocks_are_the_real_langchain_shape():
    """Verified against langchain-core 1.5.4: a model that already knows how to
    read a standard content block needs no agentstage-specific handling."""
    from langchain_core.messages.content import FileContentBlock, ImageContentBlock

    image = content_block_for(_attachment("image/png"), b"abc")
    file = content_block_for(_attachment("text/plain"), b"abc")

    assert set(image) <= set(ImageContentBlock.__annotations__)
    assert set(file) <= set(FileContentBlock.__annotations__)
