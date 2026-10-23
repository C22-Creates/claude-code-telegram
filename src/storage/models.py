"""Data models for storage."""

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, Optional

import aiosqlite


@dataclass
class ProjectThreadModel:
    """Project-thread mapping data model."""

    project_slug: str
    chat_id: int
    message_thread_id: int
    topic_name: str
    is_active: bool = True
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    id: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        for key in ["created_at", "updated_at"]:
            if data[key]:
                data[key] = data[key].isoformat()
        return data

    @classmethod
    def from_row(cls, row: aiosqlite.Row) -> "ProjectThreadModel":
        """Create from database row."""
        data = dict(row)

        for field in ["created_at", "updated_at"]:
            val = data.get(field)
            if val and isinstance(val, str):
                data[field] = datetime.fromisoformat(val)
        data["is_active"] = bool(data.get("is_active", True))

        return cls(**data)
