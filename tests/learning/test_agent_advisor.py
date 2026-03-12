"""Tests for agent advisor policy."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from openjarvis.learning.agent_advisor import AgentAdvisorPolicy


@dataclass
class _MockTrace:
    query: str = ""
    model: str = "model-a"
    outcome: str = "success"
    feedback: Optional[float] = 0.8
    steps: list = field(default_factory=list)
    total_latency_seconds: float = 1.0


class _MockTraceStore:
    def __init__(self, traces):
        self._traces = traces

    def list_traces(self):
        return self._traces


class TestAgentAdvisorPolicy:
    def test_no_problem_traces(self):
        traces = [_MockTrace(outcome="success") for _ in range(5)]
        policy = AgentAdvisorPolicy()
        store = _MockTraceStore(traces)
        result = policy.update(store)
        assert result["recommendations"] == []
        assert result["confidence"] == 1.0

    def test_detects_failures(self):
        traces = [
            _MockTrace(query="def code()", outcome="failure"),
            _MockTrace(query="def more_code()", outcome="failure"),
            _MockTrace(query="import something", outcome="failure"),
        ]
        policy = AgentAdvisorPolicy()
        store = _MockTraceStore(traces)
        result = policy.update(store)
        assert len(result["recommendations"]) > 0
        assert result["confidence"] < 1.0

    def test_detects_slow_traces(self):
        traces = [
            _MockTrace(outcome="success", total_latency_seconds=10.0),
        ]
        policy = AgentAdvisorPolicy()
        store = _MockTraceStore(traces)
        result = policy.update(store)
        assert result["problem_traces"] == 1

    def test_empty_traces(self):
        policy = AgentAdvisorPolicy()
        store = _MockTraceStore([])
        result = policy.update(store)
        assert result["recommendations"] == []

    def test_is_agent_policy(self):
        from openjarvis.learning._stubs import AgentLearningPolicy
        assert issubclass(AgentAdvisorPolicy, AgentLearningPolicy)
        assert AgentAdvisorPolicy.target == "agent"

    def test_max_traces_limit(self):
        traces = [
            _MockTrace(outcome="failure") for _ in range(100)
        ]
        policy = AgentAdvisorPolicy(max_traces=10)
        store = _MockTraceStore(traces)
        result = policy.update(store)
        # Should only analyze last 10 traces
        assert result["problem_traces"] <= 10
