"""Tests for LogHub dataset provider."""

from openjarvis.evals.datasets.loghub import LogHubDataset


class TestLogHubDataset:
    def test_instantiation(self) -> None:
        ds = LogHubDataset()
        assert ds.dataset_id == "loghub"
        assert ds.dataset_name == "LogHub"

    def test_has_required_methods(self) -> None:
        ds = LogHubDataset()
        assert hasattr(ds, "load")
        assert hasattr(ds, "iter_records")
        assert hasattr(ds, "size")
