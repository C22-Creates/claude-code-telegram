"""Entry point: ``python -m src.main`` (what systemd runs).

Builds the harness app (``src.harness.app``), starts it, and on SIGINT/SIGTERM
runs its ordered, bounded shutdown.
"""

import argparse
import asyncio
import logging
import signal
import sys
from pathlib import Path

import structlog

from src import __version__
from src.config import FeatureFlags, load_config
from src.exceptions import ConfigurationError
from src.harness.app import HarnessApp, build


def setup_logging(debug: bool = False) -> None:
    """Configure structured logging."""
    level = logging.DEBUG if debug else logging.INFO

    # Configure standard logging
    logging.basicConfig(
        level=level,
        format="%(message)s",
        stream=sys.stdout,
    )

    # httpx logs every request URL at INFO, and Telegram Bot API URLs embed the
    # bot token (/bot<token>/getUpdates), so INFO-level httpx output writes the
    # token into journald on every poll. Warnings and errors still come through.
    for noisy in ("httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(max(level, logging.WARNING))

    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            (
                structlog.processors.JSONRenderer()
                if not debug
                else structlog.dev.ConsoleRenderer()
            ),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Claude Code Telegram Bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--version", action="version", version=f"Claude Code Telegram Bot {__version__}"
    )

    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    parser.add_argument("--config-file", type=Path, help="Path to configuration file")

    return parser.parse_args()


async def run_application(app: HarnessApp) -> None:
    """Start the app and run until a shutdown signal; always stop it."""
    logger = structlog.get_logger()
    shutdown_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    def on_signal(signum: int) -> None:
        logger.info("Shutdown signal received", signal=signum)
        shutdown_event.set()

    for signum in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(signum, on_signal, signum)

    try:
        logger.info("Starting Claude Code Telegram Bot")
        await app.start()
        await shutdown_event.wait()
    except Exception as e:
        logger.error("Application error", error=str(e))
        raise
    finally:
        await app.stop()


async def main() -> None:
    """Main application entry point."""
    args = parse_args()
    setup_logging(debug=args.debug)

    logger = structlog.get_logger()
    logger.info("Starting Claude Code Telegram Bot", version=__version__)

    try:
        config = load_config(config_file=args.config_file)
        logger.info(
            "Configuration loaded",
            environment="production" if config.is_production else "development",
            enabled_features=FeatureFlags(config).get_enabled_features(),
            debug=config.debug,
        )
        app = await build(config)
        await run_application(app)

    except ConfigurationError as e:
        logger.error("Configuration error", error=str(e))
        sys.exit(1)
    except Exception as e:
        logger.exception("Unexpected error", error=str(e))
        sys.exit(1)


def run() -> None:
    """Synchronous entry point for setuptools."""
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutdown requested by user")
        sys.exit(0)


if __name__ == "__main__":
    run()
