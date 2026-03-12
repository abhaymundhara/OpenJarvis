"""Tests for the trace-driven router policy."""

from __future__ import annotations

import time
from pathlib import Path

from openjarvis.core.types import StepType, Trace, TraceStep
from openjarvis.learning._stubs import RoutingContext
from openjarvis.learning.trace_policy import TraceDrivenPolicy, classify_query
from openjarvis.traces.analyzer import TraceAnalyzer
from openjarvis.traces.store import TraceStore


def _make_trace(
    query: str = "test",
    model: str = "qwen3:8b",
    outcome: str | None = "success",
    feedback: float | None = 0.8,
) -> Trace:
    now = time.time()
    return Trace(
        query=query,
        agent="orchestrator",
        model=model,
        engine="ollama",
        result="result",
        outcome=outcome,
        feedback=feedback,
        started_at=now,
        ended_at=now + 0.5,
        total_tokens=100,
        total_latency_seconds=0.5,
        steps=[
            TraceStep(
                step_type=StepType.GENERATE,
                timestamp=now,
                duration_seconds=0.5,
                output={"tokens": 100},
            ),
        ],
    )


class TestClassifyQuery:
    def test_code(self) -> None:
        assert classify_query("def hello(): pass") == "code"
        assert classify_query("```python\nprint()```") == "code"

    def test_math(self) -> None:
        assert classify_query("solve this equation for x") == "math"
        assert classify_query("compute the integral") == "math"

    def test_short(self) -> None:
        assert classify_query("hello") == "short"
        assert classify_query("what time is it?") == "short"

    def test_long(self) -> None:
        assert classify_query("a" * 501) == "long"

    def test_general(self) -> None:
        q = "Tell me about the history of artificial intelligence research"
        assert classify_query(q) == "general"


class TestTraceDrivenPolicy:
    def test_fallback_no_traces(self) -> None:
        policy = TraceDrivenPolicy(default_model="qwen3:8b")
        ctx = RoutingContext(query="hello")
        assert policy.select_model(ctx) == "qwen3:8b"

    def test_fallback_chain(self) -> None:
        policy = TraceDrivenPolicy(
            default_model="missing",
            fallback_model="llama3:8b",
            available_models=["llama3:8b"],
        )
        ctx = RoutingContext(query="hello")
        assert policy.select_model(ctx) == "llama3:8b"

    def test_fallback_first_available(self) -> None:
        policy = TraceDrivenPolicy(available_models=["modelA", "modelB"])
        ctx = RoutingContext(query="hello")
        assert policy.select_model(ctx) == "modelA"

    def test_update_from_traces(self, tmp_path: Path) -> None:
        store = TraceStore(tmp_path / "test.db")
        # Create traces: code queries succeed more with codestral
        for _ in range(6):
            store.save(_make_trace(
                query="def foo(): pass",
                model="codestral",
                outcome="success",
                feedback=0.9,
            ))
        for _ in range(6):
            store.save(_make_trace(
                query="def bar(): return 1",
                model="qwen3:8b",
                outcome="failure",
                feedback=0.3,
            ))

        analyzer = TraceAnalyzer(store)
        policy = TraceDrivenPolicy(
            analyzer=analyzer,
            default_model="qwen3:8b",
        )
        policy.min_samples = 3

        result = policy.update_from_traces()
        assert result["updated"] is True

        # Policy should now route code to codestral
        ctx = RoutingContext(query="import os; def main(): pass")
        assert policy.select_model(ctx) == "codestral"
        store.close()

    def test_policy_map_readable(self, tmp_path: Path) -> None:
        store = TraceStore(tmp_path / "test.db")
        for _ in range(5):
            store.save(_make_trace(
                query="hello", model="small-model",
                outcome="success",
            ))

        analyzer = TraceAnalyzer(store)
        policy = TraceDrivenPolicy(analyzer=analyzer, default_model="default")
        policy.min_samples = 3
        policy.update_from_traces()

        pmap = policy.policy_map
        assert isinstance(pmap, dict)
        assert "short" in pmap
        assert pmap["short"] == "small-model"
        store.close()

    def test_respects_min_samples(self, tmp_path: Path) -> None:
        store = TraceStore(tmp_path / "test.db")
        # Only 2 traces — below threshold
        store.save(_make_trace(query="hello", model="small", outcome="success"))
        store.save(_make_trace(query="hi", model="small", outcome="success"))

        analyzer = TraceAnalyzer(store)
        policy = TraceDrivenPolicy(
            analyzer=analyzer,
            default_model="default",
        )
        policy.min_samples = 5
        policy.update_from_traces()

        ctx = RoutingContext(query="hey")
        # Should fallback since not enough confidence
        assert policy.select_model(ctx) == "default"
        store.close()

    def test_respects_available_models(self, tmp_path: Path) -> None:
        store = TraceStore(tmp_path / "test.db")
        for _ in range(10):
            store.save(_make_trace(
                query="hello", model="unavailable-model",
                outcome="success",
            ))

        analyzer = TraceAnalyzer(store)
        policy = TraceDrivenPolicy(
            analyzer=analyzer,
            available_models=["qwen3:8b", "llama3:8b"],
            default_model="qwen3:8b",
        )
        policy.min_samples = 3
        policy.update_from_traces()

        ctx = RoutingContext(query="hey")
        # Should fallback since learned model not in available_models
        assert policy.select_model(ctx) == "qwen3:8b"
        store.close()

    def test_observe_online(self) -> None:
        policy = TraceDrivenPolicy(default_model="default")
        policy.min_samples = 3

        # First observation creates the entry
        policy.observe("hello", "fast-model", "success", 0.9)
        assert policy.policy_map.get("short") == "fast-model"

        # Not enough samples yet for high confidence
        ctx = RoutingContext(query="hi")
        # Confidence is 1 < min_samples=3, so fallback
        assert policy.select_model(ctx) == "default"

    def test_update_no_analyzer(self) -> None:
        policy = TraceDrivenPolicy()
        result = policy.update_from_traces()
        assert result["error"] == "no analyzer configured"

    def test_update_empty_store(self, tmp_path: Path) -> None:
        store = TraceStore(tmp_path / "test.db")
        analyzer = TraceAnalyzer(store)
        policy = TraceDrivenPolicy(analyzer=analyzer)
        result = policy.update_from_traces()
        assert result["updated"] is False
        store.close()
