"""Feature flag management."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .settings import Settings


class FeatureFlags:
    """Derived on/off switches for the optional parts of the bot."""

    def __init__(self, settings: "Settings"):
        """Initialize with settings."""
        self.settings = settings

    @property
    def mcp_enabled(self) -> bool:
        """Check if Model Context Protocol is enabled."""
        return self.settings.enable_mcp and self.settings.mcp_config_path is not None

    @property
    def scheduler_enabled(self) -> bool:
        """Check if the job scheduler is enabled."""
        return self.settings.enable_scheduler

    @property
    def dispatcher_enabled(self) -> bool:
        """Check if the Hermes task-board dispatcher is enabled."""
        return self.settings.enable_dispatcher

    @property
    def voice_messages_enabled(self) -> bool:
        """Check if voice message transcription is enabled and usable."""
        if not self.settings.enable_voice_messages:
            return False
        if self.settings.voice_provider == "local":
            return True  # No API key needed for local whisper.cpp
        if self.settings.voice_provider == "openai":
            return self.settings.openai_api_key is not None
        return self.settings.mistral_api_key is not None

    def is_feature_enabled(self, feature_name: str) -> bool:
        """Generic feature check by name."""
        return feature_name in self.get_enabled_features()

    def get_enabled_features(self) -> list[str]:
        """Get list of all enabled features."""
        flags = {
            "mcp": self.mcp_enabled,
            "scheduler": self.scheduler_enabled,
            "dispatcher": self.dispatcher_enabled,
            "voice_messages": self.voice_messages_enabled,
        }
        return [name for name, enabled in flags.items() if enabled]
