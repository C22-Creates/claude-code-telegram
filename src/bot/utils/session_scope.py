"""Session scope derivation for Telegram conversations.

A "scope" is the boundary a Claude session belongs to. Sessions used to be
keyed by (user_id, project_path) alone, so every forum topic — and every
background job running under the same user — shared one pool of
``max_sessions_per_user`` slots. A newsletter task could evict the session
behind a topic the user was still talking in, and the next message would start
cold.

Chat scopes are ``chat:<chat_id>:<thread_id>``. Background work uses its own
scopes (see ``SCOPE_HERMES`` / ``SCOPE_SCHEDULER`` in utils.constants).
"""

from typing import Any, Optional

CHAT_SCOPE_PREFIX = "chat"


def scope_key_from_context(context: Any) -> Optional[str]:
    """Derive the session scope for the update being handled.

    Returns ``None`` outside thread mode (e.g. plain DMs), which keeps the
    legacy unscoped bucket working exactly as before.
    """
    user_data = getattr(context, "user_data", None)
    if not user_data:
        return None

    thread_context = user_data.get("_thread_context")
    if not thread_context:
        return None

    chat_id = thread_context.get("chat_id")
    thread_id = thread_context.get("message_thread_id")
    if chat_id is None or thread_id is None:
        return None

    return f"{CHAT_SCOPE_PREFIX}:{chat_id}:{thread_id}"
