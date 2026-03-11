"""AgentScheduler — cron/interval tick scheduling for managed agents."""

from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from openjarvis.agents.executor import AgentExecutor
    from openjarvis.agents.manager import AgentManager

logger = logging.getLogger(__name__)


def _next_cron_fire(cron_expr: str, now: float | None = None) -> float:
    """Calculate the next fire time for a cron expression.

    Uses croniter if available, otherwise falls back to a simple
    interval-based approximation.
    """
    try:
        from croniter import croniter
    except ImportError:
        # Fallback: treat as hourly interval
        logger.warning("croniter not installed, treating cron as 3600s interval")
        return (now or time.time()) + 3600

    base = now or time.time()
    import datetime

    dt = datetime.datetime.fromtimestamp(base)
    cron = croniter(cron_expr, dt)
    next_dt = cron.get_next(datetime.datetime)
    return next_dt.timestamp()


class AgentScheduler:
    """Schedules managed agent ticks based on cron/interval configs.

    Runs a background thread that checks for due agents and dispatches
    ticks to the executor.
    """

    def __init__(
        self,
        manager: AgentManager,
        executor: AgentExecutor | Any,
        tick_interval: float = 1.0,
    ) -> None:
        self._manager = manager
        self._executor = executor
        self._tick_interval = tick_interval
        # agent_id -> {schedule_type, schedule_value, next_fire}
        self._agents: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    @property
    def registered_agents(self) -> set[str]:
        with self._lock:
            return set(self._agents.keys())

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def register_agent(self, agent_id: str) -> None:
        """Register an agent for scheduling."""
        agent = self._manager.get_agent(agent_id)
        if agent is None:
            raise ValueError(f"Agent {agent_id} not found")

        config = agent.get("config", {})
        schedule_type = config.get("schedule_type", "manual")
        schedule_value = config.get("schedule_value", 0)

        now = time.time()
        if schedule_type == "cron":
            next_fire = _next_cron_fire(str(schedule_value), now)
        elif schedule_type == "interval":
            next_fire = now + float(schedule_value)
        else:
            next_fire = float("inf")  # Manual: never auto-fires

        with self._lock:
            self._agents[agent_id] = {
                "schedule_type": schedule_type,
                "schedule_value": schedule_value,
                "next_fire": next_fire,
            }

        logger.info(
            "Registered agent %s (%s), next fire: %s",
            agent_id,
            schedule_type,
            next_fire,
        )

    def deregister_agent(self, agent_id: str) -> None:
        """Remove an agent from scheduling."""
        with self._lock:
            self._agents.pop(agent_id, None)
        logger.info("Deregistered agent %s", agent_id)

    def start(self) -> None:
        """Start the scheduler background thread."""
        if self.is_running:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="agent-scheduler"
        )
        self._thread.start()
        logger.info("Agent scheduler started")

    def stop(self) -> None:
        """Stop the scheduler background thread."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
            self._thread = None
        logger.info("Agent scheduler stopped")

    def _loop(self) -> None:
        """Main scheduler loop."""
        while not self._stop_event.is_set():
            try:
                self._check_due_agents()
            except Exception:
                logger.exception("Scheduler tick error")
            self._stop_event.wait(self._tick_interval)

    def _check_due_agents(self) -> None:
        """Check all registered agents and fire those that are due."""
        now = time.time()

        with self._lock:
            due = [
                (aid, info)
                for aid, info in self._agents.items()
                if info["next_fire"] <= now
            ]

        for agent_id, info in due:
            agent = self._manager.get_agent(agent_id)
            if agent is None or agent["status"] in ("paused", "archived", "running"):
                continue

            logger.info("Firing tick for agent %s", agent_id)
            try:
                self._executor.execute_tick(agent_id)
            except Exception:
                logger.exception("Error executing tick for agent %s", agent_id)

            # Update next fire time
            with self._lock:
                if agent_id in self._agents:
                    if info["schedule_type"] == "cron":
                        self._agents[agent_id]["next_fire"] = _next_cron_fire(
                            str(info["schedule_value"]), now,
                        )
                    elif info["schedule_type"] == "interval":
                        self._agents[agent_id]["next_fire"] = (
                            now + float(info["schedule_value"])
                        )
                    # Manual: stays at inf
