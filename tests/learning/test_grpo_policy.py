"""Tests for GRPORouterPolicy — Group Relative Policy Optimization."""

from __future__ import annotations

import pytest

from openjarvis.core.registry import RouterPolicyRegistry
from openjarvis.core.types import RoutingContext
from openjarvis.learning.grpo_policy import (
    GRPORouterPolicy,
    GRPOSample,
    GRPOState,
    ensure_registered,
)


def _make_context(query: str = "hello", **kwargs) -> RoutingContext:
    """Build a RoutingContext with sensible defaults."""
    return RoutingContext(
        query=query,
        query_length=kwargs.pop("query_length", len(query)),
        has_code=kwargs.pop("has_code", False),
        has_math=kwargs.pop("has_math", False),
        **kwargs,
    )


class TestGRPORouterPolicy:
    def test_route_with_no_models(self) -> None:
        policy = GRPORouterPolicy()
        ctx = _make_context()
        with pytest.raises(ValueError, match="No models available"):
            policy.route(ctx, [])

    def test_route_random_before_min_samples(self) -> None:
        policy = GRPORouterPolicy(min_samples=5)
        models = ["model-a", "model-b", "model-c"]
        ctx = _make_context()
        # Before any samples, should still return a model from the list
        for _ in range(20):
            result = policy.route(ctx, models)
            assert result in models

    def test_add_sample_increments_count(self) -> None:
        policy = GRPORouterPolicy()
        assert policy.state.sample_counts.get("code", 0) == 0
        policy.add_sample("code", "model-a", 0.9)
        assert policy.state.sample_counts["code"] == 1
        policy.add_sample("code", "model-b", 0.7)
        assert policy.state.sample_counts["code"] == 2

    def test_update_no_samples(self) -> None:
        policy = GRPORouterPolicy()
        result = policy.update()
        assert result["updated"] is False
        assert result["reason"] == "no samples"

    def test_update_with_samples(self) -> None:
        policy = GRPORouterPolicy()
        policy.add_sample("code", "model-a", 0.9)
        policy.add_sample("code", "model-b", 0.3)
        policy.add_sample("math", "model-a", 0.5)
        policy.add_sample("math", "model-b", 0.8)

        result = policy.update()
        assert result["updated"] is True
        assert result["samples_processed"] == 4
        assert result["groups"] == 2
        assert result["updates_applied"] == 4
        assert result["total_updates"] == 1

    def test_update_shifts_weights(self) -> None:
        policy = GRPORouterPolicy(learning_rate=0.5)
        # Add many high-reward samples for model-a on "code"
        for _ in range(10):
            policy.add_sample("code", "model-a", 0.95)
            policy.add_sample("code", "model-b", 0.1)

        policy.update()

        # model-a should have higher weight for "code" than model-b
        weight_a = policy.state.weights.get("model-a", {}).get("code", 0.0)
        weight_b = policy.state.weights.get("model-b", {}).get("code", 0.0)
        assert weight_a > weight_b

    def test_route_after_training(self) -> None:
        policy = GRPORouterPolicy(learning_rate=1.0, min_samples=5, temperature=0.1)
        models = ["model-a", "model-b"]

        # Train: model-a is much better for short queries
        for _ in range(20):
            policy.add_sample("short", "model-a", 0.95)
            policy.add_sample("short", "model-b", 0.1)
        policy.update()

        # Route 100 times — model-a should be selected significantly more often
        ctx = _make_context("hi", query_length=5)
        counts = {"model-a": 0, "model-b": 0}
        for _ in range(100):
            selected = policy.route(ctx, models)
            counts[selected] += 1

        # model-a should win the majority of the time
        assert counts["model-a"] > 70, (
            f"model-a only selected {counts['model-a']}/100 times"
        )

    def test_reset(self) -> None:
        policy = GRPORouterPolicy()
        policy.add_sample("code", "model-a", 0.9)
        policy.add_sample("code", "model-b", 0.3)
        policy.update()

        assert policy.state.total_updates == 1
        assert len(policy.state.weights) > 0

        policy.reset()

        assert policy.state.total_updates == 0
        assert policy.state.sample_counts.get("code", 0) == 0
        # After reset, weights should be empty (fresh defaultdict)
        assert len(policy.state.weights) == 0

    def test_registry_registration(self) -> None:
        ensure_registered()
        assert RouterPolicyRegistry.contains("grpo")
        assert RouterPolicyRegistry.get("grpo") is GRPORouterPolicy

    def test_update_single_sample_group_skipped(self) -> None:
        """Groups with only 1 sample are skipped (need >= 2 for comparison)."""
        policy = GRPORouterPolicy()
        policy.add_sample("code", "model-a", 0.9)
        result = policy.update()
        # update() still runs but the single-sample group is skipped
        assert result["updated"] is True
        assert result["updates_applied"] == 0

    def test_multiple_updates_accumulate(self) -> None:
        policy = GRPORouterPolicy(learning_rate=0.1)
        for batch in range(3):
            policy.add_sample("general", "model-a", 0.8)
            policy.add_sample("general", "model-b", 0.2)
            result = policy.update()
            assert result["total_updates"] == batch + 1

        assert policy.state.total_updates == 3

    def test_grpo_sample_dataclass(self) -> None:
        sample = GRPOSample(query_class="code", model="m1", reward=0.75)
        assert sample.query_class == "code"
        assert sample.model == "m1"
        assert sample.reward == 0.75

    def test_grpo_state_dataclass(self) -> None:
        state = GRPOState()
        assert state.total_updates == 0
        assert state.sample_counts["any"] == 0
        assert state.weights["any"]["class"] == 0.0
