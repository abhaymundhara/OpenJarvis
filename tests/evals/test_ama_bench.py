"""Tests for AMA-Bench dataset provider."""

from openjarvis.evals.datasets.ama_bench import AMABenchDataset


class TestAMABenchDataset:
    def test_instantiation(self) -> None:
        ds = AMABenchDataset()
        assert ds.dataset_id == "ama-bench"
        assert ds.dataset_name == "AMA-Bench"

    def test_has_required_methods(self) -> None:
        ds = AMABenchDataset()
        assert hasattr(ds, "load")
        assert hasattr(ds, "iter_records")
        assert hasattr(ds, "size")
        assert hasattr(ds, "iter_episodes")


from unittest.mock import MagicMock
from openjarvis.evals.core.types import EvalRecord
from openjarvis.evals.scorers.ama_bench_judge import AMABenchScorer


def _mock_backend() -> MagicMock:
    backend = MagicMock()
    backend.generate.return_value = "CORRECT"
    return backend


class TestAMABenchScorer:
    def test_instantiation(self) -> None:
        s = AMABenchScorer(_mock_backend(), "test-model")
        assert s.scorer_id == "ama-bench"

    def test_empty_response(self) -> None:
        s = AMABenchScorer(_mock_backend(), "test-model")
        record = EvalRecord(
            record_id="test-1", problem="question",
            reference="answer", category="agentic",
        )
        is_correct, meta = s.score(record, "")
        assert is_correct is False
        assert meta["reason"] == "empty_response"


class TestAMABenchCLI:
    def test_in_benchmarks_dict(self) -> None:
        from openjarvis.evals.cli import BENCHMARKS
        assert "ama-bench" in BENCHMARKS

    def test_build_dataset(self) -> None:
        from openjarvis.evals.cli import _build_dataset
        ds = _build_dataset("ama-bench")
        assert ds.dataset_id == "ama-bench"

    def test_build_scorer(self) -> None:
        from openjarvis.evals.cli import _build_scorer
        s = _build_scorer("ama-bench", _mock_backend(), "test-model")
        assert s.scorer_id == "ama-bench"
