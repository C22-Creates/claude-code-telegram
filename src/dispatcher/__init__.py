"""Hermes task-board dispatcher.

Polls the durable task board (defined in the c22os repo's `hermes` package),
runs ready tasks through Claude, and reconciles their lifecycle — completing,
blocking-for-human, or failing-with-backoff. Reclaims tasks from crashed
workers via heartbeat (the circuit breaker).

Dormant unless `enable_dispatcher` is set. See src/dispatcher/service.py.
"""

from .service import DispatcherService, RunOutcome, TaskRunner

__all__ = ["DispatcherService", "RunOutcome", "TaskRunner"]
