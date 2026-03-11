"""Tests for AgentManager persistent agent lifecycle."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def manager():
    """Create an AgentManager with a temp database."""
    from openjarvis.agents.manager import AgentManager

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "agents.db"
        mgr = AgentManager(db_path=str(db_path))
        yield mgr
        mgr.close()


class TestAgentCRUD:
    def test_create_agent(self, manager):
        agent = manager.create_agent(
            name="researcher",
            agent_type="monitor_operative",
            config={
                "tools": ["web_search"],
                "schedule_type": "cron",
                "schedule_value": "0 9 * * *",
            },
        )
        assert agent["id"]
        assert agent["name"] == "researcher"
        assert agent["agent_type"] == "monitor_operative"
        assert agent["status"] == "idle"

    def test_list_agents(self, manager):
        manager.create_agent(name="agent1", agent_type="simple")
        manager.create_agent(name="agent2", agent_type="orchestrator")
        agents = manager.list_agents()
        assert len(agents) == 2
        names = {a["name"] for a in agents}
        assert names == {"agent1", "agent2"}

    def test_get_agent(self, manager):
        created = manager.create_agent(name="test", agent_type="simple")
        fetched = manager.get_agent(created["id"])
        assert fetched is not None
        assert fetched["name"] == "test"

    def test_get_agent_not_found(self, manager):
        assert manager.get_agent("nonexistent") is None

    def test_update_agent(self, manager):
        created = manager.create_agent(name="old", agent_type="simple")
        updated = manager.update_agent(created["id"], name="new")
        assert updated["name"] == "new"

    def test_delete_agent_soft(self, manager):
        created = manager.create_agent(name="doomed", agent_type="simple")
        manager.delete_agent(created["id"])
        agent = manager.get_agent(created["id"])
        assert agent["status"] == "archived"

    def test_pause_resume(self, manager):
        created = manager.create_agent(name="pausable", agent_type="simple")
        manager.pause_agent(created["id"])
        assert manager.get_agent(created["id"])["status"] == "paused"
        manager.resume_agent(created["id"])
        assert manager.get_agent(created["id"])["status"] == "idle"


class TestTaskCRUD:
    def test_create_task(self, manager):
        agent = manager.create_agent(name="worker", agent_type="simple")
        task = manager.create_task(agent["id"], description="Find papers on reasoning")
        assert task["id"]
        assert task["description"] == "Find papers on reasoning"
        assert task["status"] == "pending"

    def test_list_tasks(self, manager):
        agent = manager.create_agent(name="worker", agent_type="simple")
        manager.create_task(agent["id"], description="task1")
        manager.create_task(agent["id"], description="task2")
        tasks = manager.list_tasks(agent["id"])
        assert len(tasks) == 2

    def test_update_task(self, manager):
        agent = manager.create_agent(name="worker", agent_type="simple")
        task = manager.create_task(agent["id"], description="task1")
        updated = manager.update_task(task["id"], status="completed")
        assert updated["status"] == "completed"

    def test_delete_task(self, manager):
        agent = manager.create_agent(name="worker", agent_type="simple")
        task = manager.create_task(agent["id"], description="task1")
        manager.delete_task(task["id"])
        tasks = manager.list_tasks(agent["id"])
        assert len(tasks) == 0


class TestChannelBindings:
    def test_bind_channel(self, manager):
        agent = manager.create_agent(name="slacker", agent_type="simple")
        binding = manager.bind_channel(
            agent["id"],
            channel_type="slack",
            config={
                "channel": "#research",
                "mention_only": False,
                "typing_indicators": True,
            },
        )
        assert binding["id"]
        assert binding["channel_type"] == "slack"

    def test_list_bindings(self, manager):
        agent = manager.create_agent(name="slacker", agent_type="simple")
        manager.bind_channel(
            agent["id"], channel_type="slack", config={"channel": "#a"}
        )
        manager.bind_channel(
            agent["id"], channel_type="telegram", config={"chat_id": "123"}
        )
        bindings = manager.list_channel_bindings(agent["id"])
        assert len(bindings) == 2

    def test_unbind_channel(self, manager):
        agent = manager.create_agent(name="slacker", agent_type="simple")
        binding = manager.bind_channel(agent["id"], channel_type="slack", config={})
        manager.unbind_channel(binding["id"])
        assert len(manager.list_channel_bindings(agent["id"])) == 0


class TestSummaryMemory:
    def test_initial_summary_empty(self, manager):
        agent = manager.create_agent(name="test", agent_type="simple")
        assert agent["summary_memory"] == ""

    def test_update_summary(self, manager):
        agent = manager.create_agent(name="test", agent_type="simple")
        manager.update_summary_memory(agent["id"], "Key finding: X is Y")
        updated = manager.get_agent(agent["id"])
        assert updated["summary_memory"] == "Key finding: X is Y"

    def test_summary_max_length(self, manager):
        agent = manager.create_agent(name="test", agent_type="simple")
        long_text = "x" * 3000
        manager.update_summary_memory(agent["id"], long_text)
        updated = manager.get_agent(agent["id"])
        assert len(updated["summary_memory"]) <= 2000


class TestConcurrency:
    def test_run_tick_guard(self, manager):
        agent = manager.create_agent(name="busy", agent_type="simple")
        # Simulate agent running
        manager._set_status(agent["id"], "running")
        # Trying to run again should raise
        with pytest.raises(ValueError, match="already executing"):
            manager.start_tick(agent["id"])


class TestSchemaAndThreading:
    def test_agent_has_runtime_columns(self, manager):
        """New columns from ALTER TABLE migration should exist."""
        agent = manager.create_agent(name="test", agent_type="simple")
        assert "total_tokens" in agent
        assert "total_cost" in agent
        assert "total_runs" in agent
        assert "last_run_at" in agent
        assert "last_activity_at" in agent
        assert agent["total_tokens"] == 0
        assert agent["total_cost"] == 0.0
        assert agent["total_runs"] == 0

    def test_thread_safety(self, manager):
        """AgentManager should be usable from a different thread."""
        import threading

        results = []

        def create_in_thread():
            agent = manager.create_agent(name="threaded", agent_type="simple")
            results.append(agent)

        t = threading.Thread(target=create_in_thread)
        t.start()
        t.join(timeout=5)
        assert len(results) == 1
        assert results[0]["name"] == "threaded"
