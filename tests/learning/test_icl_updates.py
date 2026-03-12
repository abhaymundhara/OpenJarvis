"""Tests for the new ICLUpdaterPolicy versioned example database features."""

from __future__ import annotations

from openjarvis.learning.icl_updater import ICLUpdaterPolicy


class TestICLUpdaterAddExample:
    def test_add_example_quality_gate(self):
        """Examples below the quality threshold should be rejected."""
        policy = ICLUpdaterPolicy(min_score=0.7)
        accepted = policy.add_example(
            query="low quality",
            response="bad answer",
            outcome=0.3,
        )
        assert accepted is False
        assert len(policy.example_db) == 0
        assert policy.version == 0

    def test_add_example_stores(self):
        """Examples above the threshold should be stored."""
        policy = ICLUpdaterPolicy(min_score=0.5)
        accepted = policy.add_example(
            query="What is 2+2?",
            response="4",
            outcome=0.9,
            metadata={"source": "test"},
        )
        assert accepted is True
        assert len(policy.example_db) == 1
        ex = policy.example_db[0]
        assert ex["query"] == "What is 2+2?"
        assert ex["response"] == "4"
        assert ex["outcome"] == 0.9
        assert ex["metadata"] == {"source": "test"}
        assert ex["version"] == 1

    def test_add_example_at_threshold(self):
        """An example exactly at the threshold should be accepted."""
        policy = ICLUpdaterPolicy(min_score=0.7)
        accepted = policy.add_example(
            query="borderline",
            response="ok",
            outcome=0.7,
        )
        assert accepted is True
        assert len(policy.example_db) == 1


class TestICLUpdaterRollback:
    def test_rollback(self):
        """Rollback should remove examples added after the given version."""
        policy = ICLUpdaterPolicy(min_score=0.5)
        policy.add_example("q1", "r1", 0.8)
        policy.add_example("q2", "r2", 0.9)
        v2 = policy.version
        assert v2 == 2

        policy.add_example("q3", "r3", 0.85)
        policy.add_example("q4", "r4", 0.95)
        assert len(policy.example_db) == 4
        assert policy.version == 4

        policy.rollback(v2)
        assert policy.version == 2
        assert len(policy.example_db) == 2
        queries = [ex["query"] for ex in policy.example_db]
        assert "q1" in queries
        assert "q2" in queries
        assert "q3" not in queries
        assert "q4" not in queries

    def test_rollback_to_zero(self):
        """Rollback to version 0 should remove all examples."""
        policy = ICLUpdaterPolicy(min_score=0.5)
        policy.add_example("q1", "r1", 0.8)
        policy.add_example("q2", "r2", 0.9)
        policy.rollback(0)
        assert policy.version == 0
        assert len(policy.example_db) == 0


class TestICLUpdaterGetExamples:
    def test_get_examples(self):
        """Should return examples sorted by outcome."""
        policy = ICLUpdaterPolicy(min_score=0.5)
        policy.add_example("math q1", "r1", 0.7)
        policy.add_example("math q2", "r2", 0.95)
        policy.add_example("math q3", "r3", 0.8)

        results = policy.get_examples(top_k=2)
        assert len(results) == 2
        assert results[0]["outcome"] >= results[1]["outcome"]
        assert results[0]["outcome"] == 0.95

    def test_get_examples_with_query_class(self):
        """Filtering by query_class should narrow results."""
        policy = ICLUpdaterPolicy(min_score=0.5)
        policy.add_example("math problem", "42", 0.9)
        policy.add_example("code review", "looks good", 0.85)
        policy.add_example("math equation", "x=5", 0.8)

        results = policy.get_examples(query_class="math", top_k=10)
        assert len(results) == 2
        assert all("math" in ex["query"].lower() for ex in results)

    def test_get_examples_empty(self):
        """No examples should return empty list."""
        policy = ICLUpdaterPolicy(min_score=0.5)
        results = policy.get_examples(top_k=5)
        assert results == []


class TestICLUpdaterVersioning:
    def test_version_increments(self):
        """Each successful add should increment the version."""
        policy = ICLUpdaterPolicy(min_score=0.5)
        assert policy.version == 0

        policy.add_example("q1", "r1", 0.8)
        assert policy.version == 1

        policy.add_example("q2", "r2", 0.9)
        assert policy.version == 2

        # Rejected add should NOT increment version
        policy.add_example("q3", "r3", 0.1)
        assert policy.version == 2

    def test_max_examples_limit(self):
        """Adding beyond max_examples should trim the oldest entries."""
        policy = ICLUpdaterPolicy(min_score=0.5, max_examples=3)

        policy.add_example("q1", "r1", 0.8)
        policy.add_example("q2", "r2", 0.85)
        policy.add_example("q3", "r3", 0.9)
        assert len(policy.example_db) == 3

        # Adding a 4th should trim q1 (the oldest)
        policy.add_example("q4", "r4", 0.95)
        assert len(policy.example_db) == 3
        queries = [ex["query"] for ex in policy.example_db]
        assert "q1" not in queries
        assert "q4" in queries
        assert "q2" in queries
        assert "q3" in queries


class TestICLUpdaterAutoApply:
    def test_auto_apply_default_false(self):
        """auto_apply should default to False."""
        policy = ICLUpdaterPolicy()
        assert policy._auto_apply is False

    def test_auto_apply_configurable(self):
        """auto_apply can be set via constructor."""
        policy = ICLUpdaterPolicy(auto_apply=True)
        assert policy._auto_apply is True


class TestICLUpdaterBackwardCompat:
    def test_existing_update_still_works(self):
        """The original update() method must still function."""
        from dataclasses import dataclass, field
        from typing import Optional

        from openjarvis.core.types import StepType, TraceStep

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

        step = TraceStep(
            step_type=StepType.TOOL_CALL,
            timestamp=0.0,
            input={"tool": "calculator"},
            output={"result": "ok"},
            metadata={"tool_name": "calculator"},
        )
        traces = [
            _MockTrace(
                query="What is 2+2?",
                outcome="success",
                feedback=0.9,
                steps=[step],
            ),
        ]
        policy = ICLUpdaterPolicy(min_score=0.5)
        store = _MockTraceStore(traces)
        result = policy.update(store)
        assert len(result["examples"]) == 1
        assert result["examples"][0]["query"] == "What is 2+2?"
