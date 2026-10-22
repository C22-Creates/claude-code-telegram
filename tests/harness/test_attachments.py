"""Attachment helpers: safe inbox paths, image blocks, the router's prompt hook."""

import base64
from datetime import datetime
from pathlib import Path

from src.config.settings import Settings
from src.harness.attachments import (
    MAX_NAME_CHARS,
    image_block,
    inbox_path,
    sanitize_filename,
)
from src.harness.topic_router import TopicRouter
from src.harness.types import Attachment, Inbound

NOW = datetime(2026, 10, 21, 9, 30, 5)


def test_sanitize_strips_directories_and_leading_dots():
    assert sanitize_filename("../../etc/passwd") == "passwd"
    assert sanitize_filename("..\\..\\boot.ini") == "boot.ini"
    assert sanitize_filename(".env") == "env"
    assert sanitize_filename("my report (v2).pdf") == "my_report__v2_.pdf"


def test_sanitize_falls_back_and_caps_length_keeping_the_extension():
    assert sanitize_filename("") == "file"
    assert sanitize_filename("...") == "file"
    long = sanitize_filename("a" * 500 + ".xlsx")
    assert len(long) == MAX_NAME_CHARS
    assert long.endswith(".xlsx")


def test_inbox_path_is_per_topic_and_timestamped(tmp_path):
    path = inbox_path(tmp_path, "-100123:7", "../x.pdf", NOW)
    assert path == tmp_path / "-100123_7" / "20261021-093005-x.pdf"
    assert path.parent.is_dir()


def test_image_block_is_base64_of_the_file(tmp_path):
    f = tmp_path / "p.jpg"
    f.write_bytes(b"\xff\xd8jpeg")
    block = image_block(Attachment(kind="photo", path=f, mime="image/jpeg"))
    assert block == {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/jpeg",
            "data": base64.b64encode(b"\xff\xd8jpeg").decode(),
        },
    }


def _router(tmp_path) -> TopicRouter:
    settings = Settings(
        telegram_bot_token="t",
        telegram_bot_username="b",
        approved_directory=str(tmp_path),
    )
    return TopicRouter(
        None,
        None,
        None,
        settings,
        resolve_topic=lambda i: None,
        security_validator=None,
    )


def _inbound(attachments: list[Attachment], text: str = "look") -> Inbound:
    return Inbound(
        topic_key="1:0",
        chat_id=1,
        thread_id=None,
        message_id=2,
        user_id=3,
        text=text,
        attachments=attachments,
    )


def test_router_prompt_has_image_blocks_then_text_for_photos(tmp_path):
    f = tmp_path / "p.jpg"
    f.write_bytes(b"img")
    prompt = _router(tmp_path)._build_prompt(
        _inbound([Attachment(kind="photo", path=f, mime="image/jpeg")])
    )
    assert isinstance(prompt, list)
    assert prompt[0]["type"] == "image"
    assert prompt[0]["source"]["data"] == base64.b64encode(b"img").decode()
    assert prompt[1] == {"type": "text", "text": f"look\n\nAttached file: {f}"}


def test_router_prompt_for_documents_is_text_with_the_path(tmp_path):
    f = tmp_path / "d.pdf"
    f.write_bytes(b"pdf")
    prompt = _router(tmp_path)._build_prompt(
        _inbound([Attachment(kind="document", path=f, mime="application/pdf")], "")
    )
    assert prompt == f"Attached file: {f}"


def test_router_prompt_keeps_the_path_when_the_image_is_unreadable(tmp_path):
    missing = Path(tmp_path / "gone.jpg")
    prompt = _router(tmp_path)._build_prompt(
        _inbound([Attachment(kind="photo", path=missing, mime="image/jpeg")])
    )
    assert prompt == f"look\n\nAttached file: {missing}"


def test_inbox_path_never_reuses_a_taken_name(tmp_path):
    first = inbox_path(tmp_path, "1:2", "plan.pdf", NOW)
    second = inbox_path(tmp_path, "1:2", "plan.pdf", NOW)
    third = inbox_path(tmp_path, "1:2", "plan.pdf", NOW)
    assert first.name == "20261021-093005-plan.pdf"
    assert second.name == "20261021-093005-plan-1.pdf"
    assert third.name == "20261021-093005-plan-2.pdf"
    assert first.exists() and second.exists()  # reserved on creation
