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
