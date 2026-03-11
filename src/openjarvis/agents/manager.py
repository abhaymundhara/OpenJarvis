"""Persistent agent lifecycle manager.

Composition layer — stores agent state in SQLite, delegates all computation
to the five existing primitives (Intelligence, Agent, Tools, Engine, Learning).
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

_CREATE_AGENTS = """\
CREATE TABLE IF NOT EXISTS managed_agents (
    id              TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    agent_type      TEXT NOT NULL DEFAULT 'monitor_operative',
    config_json     TEXT NOT NULL DEFAULT '{}',
    status          TEXT NOT NULL DEFAULT 'idle',
    summary_memory  TEXT NOT NULL DEFAULT '',
    created_at      REAL NOT NULL,
    updated_at      REAL NOT NULL
);
"""

_CREATE_TASKS = """\
CREATE TABLE IF NOT EXISTS agent_tasks (
    id              TEXT PRIMARY KEY,
    agent_id        TEXT NOT NULL REFERENCES managed_agents(id),
    description     TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'pending',
    progress_json   TEXT NOT NULL DEFAULT '{}',
    findings_json   TEXT NOT NULL DEFAULT '[]',
    created_at      REAL NOT NULL
);
"""

_CREATE_BINDINGS = """\
CREATE TABLE IF NOT EXISTS channel_bindings (
    id              TEXT PRIMARY KEY,
    agent_id        TEXT NOT NULL REFERENCES managed_agents(id),
    channel_type    TEXT NOT NULL,
    config_json     TEXT NOT NULL DEFAULT '{}',
    session_id      TEXT,
    routing_mode    TEXT NOT NULL DEFAULT 'dedicated'
);
"""

_SUMMARY_MAX = 2000


class AgentManager:
    """Persistent agent lifecycle manager with SQLite backing."""

    def __init__(self, db_path: str) -> None:
        self._db_path = str(db_path)
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.execute(_CREATE_AGENTS)
        self._conn.execute(_CREATE_TASKS)
        self._conn.execute(_CREATE_BINDINGS)
        self._conn.commit()
        # Schema migrations for runtime columns
        _MIGRATIONS = [
            "ALTER TABLE managed_agents ADD COLUMN total_tokens INTEGER DEFAULT 0",
            "ALTER TABLE managed_agents ADD COLUMN total_cost REAL DEFAULT 0",
            "ALTER TABLE managed_agents ADD COLUMN total_runs INTEGER DEFAULT 0",
            "ALTER TABLE managed_agents ADD COLUMN last_run_at REAL",
            "ALTER TABLE managed_agents ADD COLUMN last_activity_at REAL",
        ]
        for migration in _MIGRATIONS:
            try:
                self._conn.execute(migration)
            except sqlite3.OperationalError:
                pass  # Column already exists
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    # ── Agent CRUD ────────────────────────────────────────────────

    def create_agent(
        self,
        name: str,
        agent_type: str = "monitor_operative",
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        agent_id = uuid.uuid4().hex[:12]
        now = time.time()
        config_json = json.dumps(config or {})
        self._conn.execute(
            "INSERT INTO managed_agents"
            " (id, name, agent_type, config_json,"
            " status, summary_memory, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, 'idle', '', ?, ?)",
            (agent_id, name, agent_type, config_json, now, now),
        )
        self._conn.commit()
        return self.get_agent(agent_id)  # type: ignore[return-value]

    def list_agents(self, include_archived: bool = False) -> List[Dict[str, Any]]:
        query = "SELECT * FROM managed_agents"
        if not include_archived:
            query += " WHERE status != 'archived'"
        query += " ORDER BY updated_at DESC"
        rows = self._conn.execute(query).fetchall()
        return [self._row_to_agent(r) for r in rows]

    def get_agent(self, agent_id: str) -> Optional[Dict[str, Any]]:
        row = self._conn.execute(
            "SELECT * FROM managed_agents WHERE id = ?", (agent_id,)
        ).fetchone()
        return self._row_to_agent(row) if row else None

    def update_agent(self, agent_id: str, **kwargs: Any) -> Dict[str, Any]:
        sets: List[str] = []
        vals: List[Any] = []
        for key in ("name", "agent_type", "status"):
            if key in kwargs:
                sets.append(f"{key} = ?")
                vals.append(kwargs[key])
        if "config" in kwargs:
            sets.append("config_json = ?")
            vals.append(json.dumps(kwargs["config"]))
        sets.append("updated_at = ?")
        vals.append(time.time())
        vals.append(agent_id)
        self._conn.execute(
            f"UPDATE managed_agents SET {', '.join(sets)} WHERE id = ?", vals
        )
        self._conn.commit()
        return self.get_agent(agent_id)  # type: ignore[return-value]

    def delete_agent(self, agent_id: str) -> None:
        self._set_status(agent_id, "archived")

    def pause_agent(self, agent_id: str) -> None:
        self._set_status(agent_id, "paused")

    def resume_agent(self, agent_id: str) -> None:
        self._set_status(agent_id, "idle")

    def _set_status(self, agent_id: str, status: str) -> None:
        self._conn.execute(
            "UPDATE managed_agents SET status = ?, updated_at = ? WHERE id = ?",
            (status, time.time(), agent_id),
        )
        self._conn.commit()

    # ── Tick concurrency guard ────────────────────────────────────

    def start_tick(self, agent_id: str) -> None:
        """Mark agent as running. Raises ValueError if already running."""
        agent = self.get_agent(agent_id)
        if agent and agent["status"] == "running":
            raise ValueError(f"Agent {agent_id} is already executing a tick")
        self._set_status(agent_id, "running")

    def end_tick(self, agent_id: str) -> None:
        self._set_status(agent_id, "idle")

    # ── Summary memory ────────────────────────────────────────────

    def update_summary_memory(self, agent_id: str, summary: str) -> None:
        truncated = summary[:_SUMMARY_MAX]
        self._conn.execute(
            "UPDATE managed_agents SET summary_memory = ?, updated_at = ? WHERE id = ?",
            (truncated, time.time(), agent_id),
        )
        self._conn.commit()

    # ── Task CRUD ─────────────────────────────────────────────────

    def create_task(
        self, agent_id: str, description: str, status: str = "pending"
    ) -> Dict[str, Any]:
        task_id = uuid.uuid4().hex[:12]
        now = time.time()
        self._conn.execute(
            "INSERT INTO agent_tasks (id, agent_id, description, status, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (task_id, agent_id, description, status, now),
        )
        self._conn.commit()
        return self._get_task(task_id)  # type: ignore[return-value]

    def list_tasks(
        self, agent_id: str, status: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        query = "SELECT * FROM agent_tasks WHERE agent_id = ?"
        params: List[Any] = [agent_id]
        if status:
            query += " AND status = ?"
            params.append(status)
        query += " ORDER BY created_at DESC"
        rows = self._conn.execute(query, params).fetchall()
        return [self._row_to_task(r) for r in rows]

    def update_task(self, task_id: str, **kwargs: Any) -> Dict[str, Any]:
        sets: List[str] = []
        vals: List[Any] = []
        for key in ("description", "status"):
            if key in kwargs:
                sets.append(f"{key} = ?")
                vals.append(kwargs[key])
        if "progress" in kwargs:
            sets.append("progress_json = ?")
            vals.append(json.dumps(kwargs["progress"]))
        if "findings" in kwargs:
            sets.append("findings_json = ?")
            vals.append(json.dumps(kwargs["findings"]))
        if not sets:
            return self._get_task(task_id)  # type: ignore[return-value]
        vals.append(task_id)
        self._conn.execute(
            f"UPDATE agent_tasks SET {', '.join(sets)} WHERE id = ?", vals
        )
        self._conn.commit()
        return self._get_task(task_id)  # type: ignore[return-value]

    def delete_task(self, task_id: str) -> None:
        self._conn.execute("DELETE FROM agent_tasks WHERE id = ?", (task_id,))
        self._conn.commit()

    def _get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        row = self._conn.execute(
            "SELECT * FROM agent_tasks WHERE id = ?", (task_id,)
        ).fetchone()
        return self._row_to_task(row) if row else None

    # ── Channel bindings ──────────────────────────────────────────

    def bind_channel(
        self,
        agent_id: str,
        channel_type: str,
        config: Optional[Dict[str, Any]] = None,
        routing_mode: str = "dedicated",
    ) -> Dict[str, Any]:
        binding_id = uuid.uuid4().hex[:12]
        session_id = uuid.uuid4().hex[:16]
        config_json = json.dumps(config or {})
        self._conn.execute(
            "INSERT INTO channel_bindings "
            "(id, agent_id, channel_type, config_json, session_id, routing_mode) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (binding_id, agent_id, channel_type, config_json, session_id, routing_mode),
        )
        self._conn.commit()
        return self._get_binding(binding_id)  # type: ignore[return-value]

    def list_channel_bindings(self, agent_id: str) -> List[Dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM channel_bindings WHERE agent_id = ?", (agent_id,)
        ).fetchall()
        return [self._row_to_binding(r) for r in rows]

    def unbind_channel(self, binding_id: str) -> None:
        self._conn.execute("DELETE FROM channel_bindings WHERE id = ?", (binding_id,))
        self._conn.commit()

    def _get_binding(self, binding_id: str) -> Optional[Dict[str, Any]]:
        row = self._conn.execute(
            "SELECT * FROM channel_bindings WHERE id = ?", (binding_id,)
        ).fetchone()
        return self._row_to_binding(row) if row else None

    def find_binding_for_channel(
        self, channel_type: str, channel_id: str
    ) -> Optional[Dict[str, Any]]:
        """Find a dedicated binding for a specific channel."""
        rows = self._conn.execute(
            "SELECT * FROM channel_bindings WHERE channel_type = ?",
            (channel_type,),
        ).fetchall()
        for row in rows:
            binding = self._row_to_binding(row)
            config = binding.get("config", {})
            if config.get("channel") == channel_id:
                return binding
        return None

    # ── Templates ─────────────────────────────────────────────────

    @staticmethod
    def list_templates() -> List[Dict[str, Any]]:
        """Discover built-in and user templates."""
        import importlib.resources

        try:
            import tomllib
        except ModuleNotFoundError:
            import tomli as tomllib  # type: ignore[no-redef]

        templates: List[Dict[str, Any]] = []

        # Built-in templates
        try:
            tpl_dir = importlib.resources.files("openjarvis.agents") / "templates"
            for item in tpl_dir.iterdir():
                if str(item).endswith(".toml"):
                    data = tomllib.loads(item.read_text(encoding="utf-8"))
                    tpl = data.get("template", {})
                    tpl["source"] = "built-in"
                    templates.append(tpl)
        except Exception:
            pass

        # User templates
        user_dir = Path("~/.openjarvis/templates").expanduser()
        if user_dir.is_dir():
            for f in user_dir.glob("*.toml"):
                try:
                    data = tomllib.loads(f.read_text(encoding="utf-8"))
                    tpl = data.get("template", {})
                    tpl["source"] = "user"
                    templates.append(tpl)
                except Exception:
                    pass

        return templates

    def create_from_template(
        self, template_id: str, name: str, overrides: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create an agent from a template with optional overrides."""
        templates = self.list_templates()
        tpl = next((t for t in templates if t.get("id") == template_id), None)
        if not tpl:
            raise ValueError(f"Template not found: {template_id}")
        skip = {"id", "name", "description", "source"}
        config = {k: v for k, v in tpl.items() if k not in skip}
        if overrides:
            config.update(overrides)
        agent_type = config.pop("agent_type", "monitor_operative")
        return self.create_agent(name=name, agent_type=agent_type, config=config)

    # ── Row converters ────────────────────────────────────────────

    @staticmethod
    def _row_to_agent(row: sqlite3.Row) -> Dict[str, Any]:
        config_raw = row["config_json"]
        return {
            "id": row["id"],
            "name": row["name"],
            "agent_type": row["agent_type"],
            "config": json.loads(config_raw) if config_raw else {},
            "status": row["status"],
            "summary_memory": row["summary_memory"] or "",
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "total_tokens": row["total_tokens"] or 0,
            "total_cost": row["total_cost"] or 0.0,
            "total_runs": row["total_runs"] or 0,
            "last_run_at": row["last_run_at"],
            "last_activity_at": row["last_activity_at"],
        }

    @staticmethod
    def _row_to_task(row: sqlite3.Row) -> Dict[str, Any]:
        progress_raw = row["progress_json"]
        findings_raw = row["findings_json"]
        return {
            "id": row["id"],
            "agent_id": row["agent_id"],
            "description": row["description"],
            "status": row["status"],
            "progress": json.loads(progress_raw) if progress_raw else {},
            "findings": json.loads(findings_raw) if findings_raw else [],
            "created_at": row["created_at"],
        }

    @staticmethod
    def _row_to_binding(row: sqlite3.Row) -> Dict[str, Any]:
        config_raw = row["config_json"]
        return {
            "id": row["id"],
            "agent_id": row["agent_id"],
            "channel_type": row["channel_type"],
            "config": json.loads(config_raw) if config_raw else {},
            "session_id": row["session_id"] or "",
            "routing_mode": row["routing_mode"] or "auto",
        }
