"""Tests for WorkArena++ benchmark."""

from unittest.mock import MagicMock

from openjarvis.evals.core.types import EvalRecord
from openjarvis.evals.datasets.workarena import WorkArenaDataset
from openjarvis.evals.scorers.workarena_scorer import WorkArenaScorer


def _mock_backend() -> MagicMock:
    backend = MagicMock()
    backend.generate.return_value = "CORRECT"
    return backend


class TestWorkArenaDataset:
    def test_instantiation(self) -> None:
        ds = WorkArenaDataset()
        assert ds.dataset_id == "workarena"
        assert ds.dataset_name == "WorkArena++"

    def test_has_required_methods(self) -> None:
        ds = WorkArenaDataset()
        assert hasattr(ds, "load")
        assert hasattr(ds, "iter_records")
        assert hasattr(ds, "size")


class TestWorkArenaScorer:
    def test_instantiation(self) -> None:
        s = WorkArenaScorer(_mock_backend(), "test-model")
        assert s.scorer_id == "workarena"

    def test_empty_response(self) -> None:
        s = WorkArenaScorer(_mock_backend(), "test-model")
        record = EvalRecord("wa-1", "task", "expected", "agentic")
        is_correct, meta = s.score(record, "")
        assert is_correct is False
        assert meta["reason"] == "empty_response"


class TestWorkArenaCLI:
    def test_in_benchmarks(self) -> None:
        from openjarvis.evals.cli import BENCHMARKS
        assert "workarena" in BENCHMARKS

    def test_build_dataset(self) -> None:
        from openjarvis.evals.cli import _build_dataset
        ds = _build_dataset("workarena")
        assert ds.dataset_id == "workarena"

    def test_build_scorer(self) -> None:
        from openjarvis.evals.cli import _build_scorer
        s = _build_scorer("workarena", _mock_backend(), "test-model")
        assert s.scorer_id == "workarena"
