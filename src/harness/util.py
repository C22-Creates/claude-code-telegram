"""Small formatting helpers for harness output."""

from urllib.parse import urlparse, urlunparse

from src.harness.types import Outcome, RunResult


def fmt_elapsed(seconds: float) -> str:
    """Format a duration as ``4m12s``, or ``42s`` under a minute."""
    total = int(seconds)
    minutes, secs = divmod(total, 60)
    return f"{minutes}m{secs}s" if minutes else f"{secs}s"


def footer(result: RunResult) -> str:
    """Outcome footer appended to the final message of a run."""
    elapsed = fmt_elapsed(result.elapsed_s)
    if result.outcome is Outcome.DONE:
        return f"✅ done · {elapsed}"
    if result.outcome is Outcome.STOPPED:
        return f"⚠️ stopped ({result.reason}) · {elapsed}"
    first_line = result.reason.splitlines()[0] if result.reason.strip() else ""
    return f"❌ error: {first_line}"


def redact_proxy_url(proxy_url: str) -> str:
    """Mask a proxy URL's password so it is safe to log."""
    try:
        parsed = urlparse(proxy_url)
    except ValueError:
        return "<unparsable proxy URL>"
    if not parsed.password:
        return proxy_url
    userinfo, _, hostport = parsed.netloc.rpartition("@")
    username = userinfo.split(":", 1)[0]
    return urlunparse(parsed._replace(netloc=f"{username}:***@{hostport}"))
