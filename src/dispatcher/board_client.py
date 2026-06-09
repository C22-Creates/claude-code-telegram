"""Load the Hermes TaskBoard from the c22os repo without vendoring it.

The board (`hermes/board.py`) is the single source of truth and lives in the
c22os repo, not here. We import it by file path so the bot never carries a
copy that could drift out of sync. The board is pure stdlib, so loading it
this way pulls in no extra dependencies.
"""

import importlib.util
from pathlib import Path
from typing import Any


def load_task_board(hermes_path: Path, db_path: Path) -> Any:
    """Import `hermes/board.py` by path and open a TaskBoard on `db_path`.

    Args:
        hermes_path: Directory containing the Hermes package (board.py, cli.py).
        db_path: Path to the SQLite board database (created if absent).

    Returns:
        An open `TaskBoard` instance.

    Raises:
        FileNotFoundError: if board.py is not found under hermes_path.
        ImportError: if the module cannot be loaded.
    """
    board_file = Path(hermes_path) / "board.py"
    if not board_file.is_file():
        raise FileNotFoundError(f"Hermes board not found at {board_file}")

    spec = importlib.util.spec_from_file_location("hermes_board", board_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load Hermes board from {board_file}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.TaskBoard(str(db_path))
