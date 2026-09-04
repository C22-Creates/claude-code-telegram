"""Load agent identity + tool enforcement from agents/<assignee>/.

The contract lives in the c22os repo at agents/README.md ("Loader contract"):
for a Hermes task whose assignee matches a directory under <working_dir>/agents/,
the identity block (SOUL + RULES + DUTIES + memory/notes.md) is prepended to the
task prompt, and agent.yaml's enforcement.disallowed_tools become hard SDK tool
denials. Unknown assignees degrade to generic behavior — a missing agent dir
must never fail a task.
"""

from pathlib import Path
from typing import List, Optional, Tuple

import structlog
import yaml

logger = structlog.get_logger()

IDENTITY_FILES = ("SOUL.md", "RULES.md", "DUTIES.md")
MEMORY_FILE = "memory/notes.md"
# Safety valve: an identity block bigger than this indicates a runaway memory
# file or a mis-pointed directory, not a legitimate agent definition.
MAX_IDENTITY_CHARS = 60_000


def load_agent(
    assignee: Optional[str], working_directory: Path
) -> Tuple[Optional[str], List[str]]:
    """Return (identity_block, disallowed_tools) for an assignee.

    (None, []) when the assignee is empty, unsafe, or has no agent directory.
    Never raises.
    """
    if not assignee:
        return None, []
    name = str(assignee).strip()
    # Assignee is free text on the board; refuse anything path-shaped.
    if not name or "/" in name or "\\" in name or name.startswith("."):
        return None, []

    agent_dir = Path(working_directory) / "agents" / name
    if not agent_dir.is_dir():
        logger.warning(
            "Assignee has no agent directory; running generic",
            assignee=name,
            agent_dir=str(agent_dir),
        )
        return None, []

    identity: Optional[str] = None
    disallowed: List[str] = []
    try:
        sections: List[str] = []
        for fname in IDENTITY_FILES:
            path = agent_dir / fname
            if path.is_file():
                text = path.read_text(encoding="utf-8").strip()
                if text:
                    sections.append(text)
        memory_path = agent_dir / MEMORY_FILE
        if memory_path.is_file():
            memory_text = memory_path.read_text(encoding="utf-8").strip()
            if memory_text:
                sections.append(
                    f"## Accumulated memory (agents/{name}/{MEMORY_FILE})\n\n"
                    f"{memory_text}"
                )

        if sections:
            identity = (
                f"# You are the `{name}` agent\n\n"
                f"The identity, constraints, and memory below are loaded from "
                f"`agents/{name}/` and are binding for this task.\n\n---\n\n"
                + "\n\n---\n\n".join(sections)
            )
            if len(identity) > MAX_IDENTITY_CHARS:
                identity = (
                    identity[:MAX_IDENTITY_CHARS] + "\n\n[identity block truncated]"
                )
                logger.warning("Agent identity block truncated", assignee=name)

        manifest = agent_dir / "agent.yaml"
        if manifest.is_file():
            data = yaml.safe_load(manifest.read_text(encoding="utf-8")) or {}
            raw = (data.get("enforcement") or {}).get("disallowed_tools") or []
            disallowed = [str(t).strip() for t in raw if str(t).strip()]
    except Exception as exc:  # noqa: BLE001 — loading must never fail the task
        logger.warning(
            "Agent load failed; running generic", assignee=name, error=str(exc)
        )
        return None, []

    logger.info(
        "Loaded agent definition",
        assignee=name,
        identity_chars=len(identity or ""),
        disallowed_tools=disallowed,
    )
    return identity, disallowed
