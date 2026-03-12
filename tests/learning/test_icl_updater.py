"""Tests for ICL updater policy — example extraction + skill discovery."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from openjarvis.core.types import StepType, TraceStep
from openjarvis.learning.icl_updater import ICLUpdaterPolicy


@dataclass
class _MockTrace:
    query: str = ""
    model: str = "model-a"
    outcome: str = "success"
    feedback: Optional[float] = 0.8
    steps: list = field(default_factory=list)
    total_latency_seconds: float = 1.0


def _make_tool_step(tool_name: str) -> TraceStep:
    return TraceStep(
        step_type=StepType.TOOL_CALL,
        timestamp=0.0,
        input={"tool": tool_name},
        output={"result": "ok"},
        metadata={"tool_name": tool_name},
    )


class _MockTraceStore:
    def __init__(self, traces):
        self._traces = traces

    def list_traces(self):
        return self._traces


class TestICLUpdaterPolicy:
    def test_empty_traces(self):
        policy = ICLUpdaterPolicy()
        store = _MockTraceStore([])
        result = policy.update(store)
        assert result["examples"] == []
        assert result["skills"] == []

    def test_extracts_examples(self):
        traces = [
            _MockTrace(
                query="What is 2+2?",
                outcome="success",
                feedback=0.9,
                steps=[_make_tool_step("calculator")],
            ),
        ]
        policy = ICLUpdaterPolicy(min_score=0.5)
        store = _MockTraceStore(traces)
        result = policy.update(store)
        assert len(result["examples"]) == 1
        assert result["examples"][0]["query"] == "What is 2+2?"

    def test_filters_low_score(self):
        traces = [
            _MockTrace(
                query="test",
                outcome="success",
                feedback=0.3,
                steps=[_make_tool_step("calculator")],
            ),
        ]
        policy = ICLUpdaterPolicy(min_score=0.7)
        store = _MockTraceStore(traces)
        result = policy.update(store)
        assert len(result["examples"]) == 0

    def test_filters_failures(self):
        traces = [
            _MockTrace(
                outcome="failure",
                feedback=0.9,
                steps=[_make_tool_step("calculator")],
            ),
        ]
        policy = ICLUpdaterPolicy()
        store = _MockTraceStore(traces)
        result = policy.update(store)
        assert len(result["examples"]) == 0

    def test_discovers_skills(self):
        # Create multiple traces with same tool sequence
        traces = [
            _MockTrace(
                query=f"Query {i}",
                outcome="success",
                feedback=0.9,
                steps=[_make_tool_step("retrieval"), _make_tool_step("llm")],
            )
            for i in range(5)
        ]
        policy = ICLUpdaterPolicy(min_score=0.5, min_skill_occurrences=3)
        store = _MockTraceStore(traces)
        result = policy.update(store)
        assert len(result["skills"]) > 0
        skill = result["skills"][0]
        assert "retrieval" in skill["sequence"]
        assert "llm" in skill["sequence"]
        assert skill["occurrences"] >= 3

    def test_max_examples(self):
        traces = [
            _MockTrace(
                query=f"Query {i}",
                outcome="success",
                feedback=0.9,
                steps=[_make_tool_step("calculator")],
            )
            for i in range(30)
        ]
        policy = ICLUpdaterPolicy(min_score=0.5, max_examples=5)
        store = _MockTraceStore(traces)
        result = policy.update(store)
        assert len(result["examples"]) <= 5

    def test_is_agent_policy(self):
        from openjarvis.learning._stubs import AgentLearningPolicy
        assert issubclass(ICLUpdaterPolicy, AgentLearningPolicy)
        assert ICLUpdaterPolicy.target == "agent"

    def test_examples_property(self):
        policy = ICLUpdaterPolicy()
        assert policy.examples == []

    def test_skills_property(self):
        policy = ICLUpdaterPolicy()
        assert policy.skills == []
