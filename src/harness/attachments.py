"""Where user attachments land on disk, and how photos reach the model.

Files are saved under ``data/inbox/<topic>/<timestamp>-<name>`` so the agent
can open them by path and Carl can find them later. Names come from Telegram
(i.e. from whoever sent the file), so they are reduced to a safe basename
before they touch the filesystem.
"""

import base64
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from src.harness.types import Attachment

MAX_NAME_CHARS = 100
_MAX_SUFFIX_CHARS = 16
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]")
FALLBACK_NAME = "file"


def sanitize_filename(name: str) -> str:
    """A safe basename: no directories, no leading dots, bounded length."""
    base = re.split(r"[/\\]", name or "")[-1]
    safe = _UNSAFE.sub("_", base).lstrip(".")
    if not safe.strip("._"):
        return FALLBACK_NAME
    if len(safe) <= MAX_NAME_CHARS:
        return safe
    suffix = Path(safe).suffix[:_MAX_SUFFIX_CHARS]
    return safe[: MAX_NAME_CHARS - len(suffix)] + suffix


def inbox_path(root: Path, topic_key: str, name: str, now: datetime) -> Path:
    """``root/<topic_key with ':'→'_'>/<YYYYmmdd-HHMMSS>-<safe name>``; dirs made."""
    folder = root / topic_key.replace(":", "_")
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{now:%Y%m%d-%H%M%S}-{sanitize_filename(name)}"


def image_block(attachment: Attachment) -> dict[str, Any]:
    """A base64 image content block (raises OSError if the file is unreadable).

    Same shape the old ``_multimodal_prompt`` sent to ``client.query()``.
    """
    data = base64.b64encode(attachment.path.read_bytes()).decode("ascii")
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": attachment.mime, "data": data},
    }
