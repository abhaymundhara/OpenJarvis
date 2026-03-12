"""Tests for SFT policy — learning from traces."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from openjarvis.learning.sft_policy import SFTRouterPolicy


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


class TestSFTRouterPolicy:
    def test_empty_traces(self):
        policy = SFTRouterPolicy()
        store = _MockTraceStore([])
        result = policy.update(store)
        assert result["updated"] is False

    def test_learns_from_traces(self):
        traces = [
            _MockTrace(
                query=f"def foo{i}(): pass",
                model="code-model", outcome="success",
                feedback=0.9,
            )
            for i in range(6)
        ]
        policy = SFTRouterPolicy(min_samples=5)
        store = _MockTraceStore(traces)
        result = policy.update(store)
        assert result["updated"] is True
        assert "code" in result["policy_map"]
        assert result["policy_map"]["code"] == "code-model"

    def test_min_samples_threshold(self):
        traces = [
            _MockTrace(query="def foo(): pass", model="code-model", outcome="success")
            for _ in range(3)
        ]
        policy = SFTRouterPolicy(min_samples=5)
        store = _MockTraceStore(traces)
        result = policy.update(store)
        assert result["updated"] is False

    def test_classify_code(self):
        assert SFTRouterPolicy._classify_query("def hello(): pass") == "code"

    def test_classify_math(self):
        assert SFTRouterPolicy._classify_query("solve the integral") == "math"

    def test_classify_short(self):
        assert SFTRouterPolicy._classify_query("hello world") == "short"

    def test_classify_general(self):
        query = "tell me about " + " ".join(["something"] * 20)
        assert SFTRouterPolicy._classify_query(query) == "general"

    def test_policy_map_property(self):
        policy = SFTRouterPolicy()
        assert policy.policy_map == {}

    def test_is_intelligence_policy(self):
        from openjarvis.learning._stubs import IntelligenceLearningPolicy
        assert issubclass(SFTRouterPolicy, IntelligenceLearningPolicy)
        assert SFTRouterPolicy.target == "intelligence"
