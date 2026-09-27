"""Claude Code integration module."""

from .exceptions import (
    ClaudeError,
    ClaudeParsingError,
    ClaudeProcessError,
    ClaudeSessionError,
    ClaudeStreamStalledError,
    ClaudeTimeoutError,
)
from .facade import ClaudeIntegration
from .sdk_integration import ClaudeResponse, ClaudeSDKManager, StreamUpdate
from .session import (
    ClaudeSession,
    SessionManager,
    SessionStorage,
)

__all__ = [
    # Exceptions
    "ClaudeError",
    "ClaudeParsingError",
    "ClaudeProcessError",
    "ClaudeSessionError",
    "ClaudeStreamStalledError",
    "ClaudeTimeoutError",
    # Main integration
    "ClaudeIntegration",
    # Core components
    "ClaudeSDKManager",
    "ClaudeResponse",
    "StreamUpdate",
    "SessionManager",
    "SessionStorage",
    "ClaudeSession",
]
