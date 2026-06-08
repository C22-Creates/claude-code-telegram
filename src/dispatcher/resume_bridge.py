"""Telegram reply -> board.resume bridge (piece 3 of the Hermes adoption).

When a task blocks for human input, the dispatcher posts the question into the
originating chat/topic with a `(task <shortid>)` marker. This bridge watches
inbound messages and, when one answers a blocked task, resumes that task with
the reply as the human's note — then short-circuits normal Claude routing so
the answer isn't also processed as a fresh prompt.

Matching, in priority order (never guesses):
  1. A `(task <shortid>)` marker in the *quoted* message (i.e. the user used
     Telegram's reply-to on the block notification) — the precise path.
  2. A `task <shortid>` / `task:<shortid>` marker typed in the reply itself.
  3. Exactly one blocked task for this (chat, topic) — the natural "just answer
     in the topic" path.
Zero or multiple candidates with no marker => not a resume; fall through.

The board is duck-typed; only `list_tasks(status=...)` and `resume(id, note=)`
are used. The bridge NEVER raises into the message handler.
"""

import re
from typing import Any, Dict, List, Optional

import structlog

logger = structlog.get_logger()

BLOCKED = "blocked"

# Matches the dispatcher's "(task abc12345)" marker, or "task: abc12345".
_TASK_MARKER = re.compile(r"task[:\s]+([0-9a-fA-F]{6,32})")


class ResumeBridge:
    """Resumes blocked board tasks from Telegram replies."""

    def __init__(self, board: Any) -> None:
        self._board = board

    async def try_resume(self, update: Any) -> bool:
        """Returns True if the message resumed a blocked task (caller stops)."""
        try:
            message = getattr(update, "message", None)
            if message is None or not getattr(message, "text", None):
                return False

            chat_id = message.chat_id
            thread_id = getattr(message, "message_thread_id", None)
            text = message.text.strip()
            quoted = ""
            reply_to = getattr(message, "reply_to_message", None)
            if reply_to is not None and getattr(reply_to, "text", None):
                quoted = reply_to.text

            candidates = self._blocked_for(chat_id, thread_id)
            if not candidates:
                return False

            task = self._match(candidates, text, quoted)
            if task is None:
                return False

            self._board.resume(task["id"], note=text)
            logger.info(
                "Resumed blocked task from reply",
                task_id=task["id"],
                chat_id=chat_id,
                thread_id=thread_id,
            )
            try:
                await message.reply_text(f"↩️ Resuming: {task.get('title', task['id'])}")
            except Exception:
                logger.warning("Resume confirmation reply failed", task_id=task["id"])
            return True
        except Exception:
            logger.exception("Resume bridge error; falling through")
            return False

    # ------------------------------------------------------------------ internals
    def _blocked_for(
        self, chat_id: int, thread_id: Optional[int]
    ) -> List[Dict[str, Any]]:
        """Blocked tasks whose delivery target matches this (chat, topic)."""
        out: List[Dict[str, Any]] = []
        for task in self._board.list_tasks(status=BLOCKED):
            task_chat = self._as_int(task.get("chat_id"))
            if task_chat is None or task_chat != chat_id:
                continue
            task_thread = None
            payload = task.get("payload")
            if isinstance(payload, dict) and payload.get("_thread_id") is not None:
                task_thread = self._as_int(payload["_thread_id"])
            if (thread_id or None) != (task_thread or None):
                continue
            out.append(task)
        return out

    def _match(
        self, candidates: List[Dict[str, Any]], text: str, quoted: str
    ) -> Optional[Dict[str, Any]]:
        # 1 & 2: explicit task marker (prefer the quoted/reply-to source).
        for source in (quoted, text):
            match = _TASK_MARKER.search(source or "")
            if not match:
                continue
            short_id = match.group(1).lower()
            for task in candidates:
                if task["id"].lower().startswith(short_id):
                    return task
        # 3: unambiguous single blocked task in this chat/topic.
        if len(candidates) == 1:
            return candidates[0]
        return None

    @staticmethod
    def _as_int(value: Any) -> Optional[int]:
        if value in (None, "", 0, "0"):
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
