"""File attachments: an in-memory store, referenced by ID rather than by value.

Security requirement (Milestone 0): do not store large binary results directly in
session state. A chat message that carried a file's bytes inline would violate
that the moment it touched an event or a checkpoint. Instead, an upload gets an
opaque ID; the chat request carries only that ID; the adapter resolves it to
bytes just before building the graph input, and the bytes never appear in an
``AgentEvent``.

The default store is in-memory and TTL'd — adequate for local development and the
examples. A production deployment should supply a store backed by object storage;
:class:`AttachmentStore` is the interface to implement.
"""

from __future__ import annotations

import base64
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from agentstage.errors import ConfigError

__all__ = [
    "Attachment",
    "AttachmentStore",
    "InMemoryAttachmentStore",
    "content_block_for",
]

#: Hard ceiling on a single upload. Not configurable per-request: a client-supplied
#: limit would let a request raise its own quota.
MAX_ATTACHMENT_BYTES = 20 * 1024 * 1024

#: How long an uploaded file is retrievable before it is evicted.
DEFAULT_TTL_SECONDS = 3600

#: File types accepted by default. An explicit allowlist, not a denylist — new
#: dangerous types appear more often than new safe ones.
DEFAULT_ALLOWED_CONTENT_TYPES: frozenset[str] = frozenset(
    {
        "text/plain",
        "text/csv",
        "text/markdown",
        "application/pdf",
        "application/json",
        "image/png",
        "image/jpeg",
        "image/webp",
        "image/gif",
    }
)


@dataclass(frozen=True, slots=True)
class Attachment:
    """A stored upload. Bytes are not part of this — see :meth:`AttachmentStore.read`.

    ``owner`` is set from whatever ``authenticate`` returned for the request that
    uploaded it, so a store can enforce that only the uploader may reference the
    file in a later chat turn. It is opaque to agentstage.
    """

    id: str
    filename: str
    content_type: str
    size: int
    created_at: float
    owner: str | None = None


@runtime_checkable
class AttachmentStore(Protocol):
    """Where uploaded files live between the upload request and the chat turn
    that references them.

    A protocol, not a base class, so a production deployment backed by S3/GCS can
    implement it without importing agentstage internals.
    """

    async def save(
        self, *, filename: str, content_type: str, data: bytes, owner: str | None
    ) -> Attachment: ...

    async def read(self, attachment_id: str, *, owner: str | None) -> tuple[Attachment, bytes]:
        """Fetch an attachment's bytes.

        Raises :class:`~agentstage.errors.ConfigError`-derived errors (via the
        implementation) if the id is unknown, expired, or ``owner`` does not match
        the one that uploaded it — the last check is what stops one user's chat
        turn from reading another user's upload by guessing an ID.
        """
        ...


class AttachmentNotFoundError(ConfigError):
    """Referenced an attachment that does not exist, expired, or isn't owned by the caller."""


class AttachmentRejectedError(ConfigError):
    """An upload failed validation: too large, disallowed type, or content that
    does not match its declared type."""


# Magic-byte signatures for the binary types in the default allowlist. A client
# fully controls the declared Content-Type header, so it is verified rather than
# trusted — this is what stops a renamed executable from riding in as "image/png".
# Plain-text types (text/*, application/json) have no reliable signature and are
# not checked here; they are still capped by size and MIME allowlist.
_MAGIC_BYTES: dict[str, tuple[bytes, ...]] = {
    "image/png": (b"\x89PNG\r\n\x1a\n",),
    "image/jpeg": (b"\xff\xd8\xff",),
    "image/gif": (b"GIF87a", b"GIF89a"),
    "image/webp": (b"RIFF",),  # followed by size then "WEBP"; prefix is enough here
    "application/pdf": (b"%PDF-",),
}


def _matches_declared_type(content_type: str, data: bytes) -> bool:
    signatures = _MAGIC_BYTES.get(content_type)
    if signatures is None:
        return True  # No signature to check against; MIME allowlist is the gate.
    return any(data.startswith(sig) for sig in signatures)


@dataclass
class InMemoryAttachmentStore:
    """The default store: process memory, TTL eviction, an owner check on read.

    Not durable across restarts and not shared across processes — adequate for
    local development and the examples, not for a multi-worker production
    deployment. Swap in an :class:`AttachmentStore` backed by real storage there.
    """

    ttl_seconds: float = DEFAULT_TTL_SECONDS
    max_bytes: int = MAX_ATTACHMENT_BYTES
    allowed_content_types: frozenset[str] = field(
        default_factory=lambda: DEFAULT_ALLOWED_CONTENT_TYPES
    )
    _files: dict[str, tuple[Attachment, bytes]] = field(default_factory=dict)

    async def save(
        self, *, filename: str, content_type: str, data: bytes, owner: str | None
    ) -> Attachment:
        self._evict_expired()

        if len(data) > self.max_bytes:
            msg = (
                f"Attachment {filename!r} is {len(data)} bytes, exceeding the "
                f"{self.max_bytes}-byte limit."
            )
            raise AttachmentRejectedError(msg)
        if content_type not in self.allowed_content_types:
            allowed = ", ".join(sorted(self.allowed_content_types))
            msg = f"Content type {content_type!r} is not allowed. Allowed types: {allowed}."
            raise AttachmentRejectedError(msg)
        if not _matches_declared_type(content_type, data):
            msg = (
                f"The file's content does not match its declared type {content_type!r}. "
                "The Content-Type header is client-supplied and was not trusted."
            )
            raise AttachmentRejectedError(msg)

        attachment = Attachment(
            id=f"file-{uuid.uuid4().hex[:16]}",
            filename=_safe_filename(filename),
            content_type=content_type,
            size=len(data),
            created_at=time.monotonic(),
            owner=owner,
        )
        self._files[attachment.id] = (attachment, data)
        return attachment

    async def read(self, attachment_id: str, *, owner: str | None) -> tuple[Attachment, bytes]:
        self._evict_expired()

        entry = self._files.get(attachment_id)
        if entry is None:
            msg = f"Attachment {attachment_id!r} was not found. It may have expired."
            raise AttachmentNotFoundError(msg)
        attachment, data = entry
        if attachment.owner != owner:
            # Same message as "not found" — confirming existence to the wrong
            # owner is itself an information leak.
            msg = f"Attachment {attachment_id!r} was not found. It may have expired."
            raise AttachmentNotFoundError(msg)
        return attachment, data

    def _evict_expired(self) -> None:
        now = time.monotonic()
        # Age-based, not a cutoff compared against created_at: with ttl_seconds=0
        # an item must expire immediately, and a cutoff of `now - 0` compared with
        # `<` requires measurable time to have passed — not reliably true on a
        # fast clock, and not what "TTL of zero" should mean.
        expired = [
            aid for aid, (att, _) in self._files.items() if now - att.created_at >= self.ttl_seconds
        ]
        for aid in expired:
            del self._files[aid]


#: Image types get an ImageContentBlock so a vision-capable model can see them;
#: everything else in the allowlist becomes a generic FileContentBlock.
_IMAGE_CONTENT_TYPES: frozenset[str] = frozenset(
    {"image/png", "image/jpeg", "image/webp", "image/gif"}
)


def content_block_for(attachment: Attachment, data: bytes) -> dict[str, Any]:
    """Build the LangChain content block for an attachment's bytes.

    Uses the real, standard shapes from ``langchain_core.messages.content``
    (``create_image_block``/``create_file_block``, verified against 1.5.4) rather
    than a bespoke agentstage format — a model that already knows how to read
    LangChain's block shape needs no adapter-specific handling to see the file.
    """
    from langchain_core.messages.content import create_file_block, create_image_block

    encoded = base64.b64encode(data).decode("ascii")
    if attachment.content_type in _IMAGE_CONTENT_TYPES:
        return dict(create_image_block(base64=encoded, mime_type=attachment.content_type))
    # FileContentBlock has no dedicated filename field; `extras` is documented as
    # the place for provider/consumer metadata that isn't the file data itself.
    return dict(
        create_file_block(
            base64=encoded, mime_type=attachment.content_type, filename=attachment.filename
        )
    )


def _safe_filename(filename: str) -> str:
    """Strip path components and control characters from a client-supplied name.

    The name is only ever displayed and echoed back, never used to build a
    filesystem path, but a name containing "../" or a null byte has no legitimate
    reason to reach a UI either.
    """
    name = filename.replace("\\", "/").rsplit("/", 1)[-1]
    name = "".join(ch for ch in name if ch.isprintable())
    return name[:255] or "upload"
