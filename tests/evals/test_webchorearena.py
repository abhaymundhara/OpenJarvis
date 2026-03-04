"""Tests for WebChoreArena benchmark."""

from unittest.mock import MagicMock

from openjarvis.evals.core.types import EvalRecord
from openjarvis.evals.datasets.webchorearena import WebChoreArenaDataset
from openjarvis.evals.scorers.webchorearena_scorer import WebChoreArenaScorer


def _mock_backend() -> MagicMock:
    backend = MagicMock()
    backend.generate.return_value = "CORRECT"
    return backend


class TestWebChoreArenaDataset:
    def test_instantiation(self) -> None:
        ds = WebChoreArenaDataset()
        assert ds.dataset_id == "webchorearena"
        assert ds.dataset_name == "WebChoreArena"

    def test_has_required_methods(self) -> None:
        ds = WebChoreArenaDataset()
        assert hasattr(ds, "load")
        assert hasattr(ds, "iter_records")
        assert hasattr(ds, "size")


class TestWebChoreArenaScorer:
    def test_instantiation(self) -> None:
        s = WebChoreArenaScorer(_mock_backend(), "test-model")
        assert s.scorer_id == "webchorearena"

    def test_string_match_correct(self) -> None:
        s = WebChoreArenaScorer(_mock_backend(), "test-model")
        record = EvalRecord(
            record_id="wca-1", problem="task",
            reference="expected answer", category="agentic",
            metadata={"eval_type": "string_match"},
        )
        is_correct, meta = s.score(record, "expected answer")
        assert is_correct is True

    def test_string_match_wrong(self) -> None:
        s = WebChoreArenaScorer(_mock_backend(), "test-model")
        record = EvalRecord(
            record_id="wca-2", problem="task",
            reference="expected answer", category="agentic",
            metadata={"eval_type": "string_match"},
        )
        is_correct, meta = s.score(record, "wrong answer")
        assert is_correct is False

    def test_empty_response(self) -> None:
        s = WebChoreArenaScorer(_mock_backend(), "test-model")
        record = EvalRecord(
            record_id="wca-3", problem="task",
            reference="answer", category="agentic",
        )
        is_correct, meta = s.score(record, "")
        assert is_correct is False


class TestWebChoreArenaCLI:
    def test_in_benchmarks(self) -> None:
        from openjarvis.evals.cli import BENCHMARKS
        assert "webchorearena" in BENCHMARKS

    def test_build_dataset(self) -> None:
        from openjarvis.evals.cli import _build_dataset
        ds = _build_dataset("webchorearena")
        assert ds.dataset_id == "webchorearena"

    def test_build_scorer(self) -> None:
        from openjarvis.evals.cli import _build_scorer
        s = _build_scorer("webchorearena", _mock_backend(), "test-model")
        assert s.scorer_id == "webchorearena"
