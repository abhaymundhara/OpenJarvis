"""Tests for LifelongAgentBench benchmark."""

from unittest.mock import MagicMock

from openjarvis.evals.core.types import EvalRecord
from openjarvis.evals.datasets.lifelong_agent import LifelongAgentDataset
from openjarvis.evals.scorers.lifelong_agent_scorer import LifelongAgentScorer


def _mock_backend() -> MagicMock:
    backend = MagicMock()
    backend.generate.return_value = "CORRECT"
    return backend


class TestLifelongAgentDataset:
    def test_instantiation(self) -> None:
        ds = LifelongAgentDataset()
        assert ds.dataset_id == "lifelong-agent"
        assert ds.dataset_name == "LifelongAgentBench"

    def test_has_episode_support(self) -> None:
        ds = LifelongAgentDataset()
        assert hasattr(ds, "iter_episodes")


class TestLifelongAgentScorer:
    def test_instantiation(self) -> None:
        s = LifelongAgentScorer(_mock_backend(), "test-model")
        assert s.scorer_id == "lifelong-agent"

    def test_empty_response(self) -> None:
        s = LifelongAgentScorer(_mock_backend(), "test-model")
        record = EvalRecord("t-1", "task", "expected", "agentic")
        is_correct, meta = s.score(record, "")
        assert is_correct is False


class TestLifelongAgentCLI:
    def test_in_benchmarks(self) -> None:
        from openjarvis.evals.cli import BENCHMARKS
        assert "lifelong-agent" in BENCHMARKS

    def test_build_dataset(self) -> None:
        from openjarvis.evals.cli import _build_dataset
        ds = _build_dataset("lifelong-agent")
        assert ds.dataset_id == "lifelong-agent"

    def test_build_scorer(self) -> None:
        from openjarvis.evals.cli import _build_scorer
        s = _build_scorer("lifelong-agent", _mock_backend(), "test-model")
        assert s.scorer_id == "lifelong-agent"
