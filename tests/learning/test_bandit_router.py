"""Tests for BanditRouterPolicy — Thompson Sampling / UCB."""

from __future__ import annotations

import pytest

from openjarvis.core.registry import RouterPolicyRegistry
from openjarvis.core.types import RoutingContext
from openjarvis.learning.bandit_router import (
    ArmStats,
    BanditRouterPolicy,
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


class TestBanditRouterPolicy:
    def test_route_no_models(self) -> None:
        policy = BanditRouterPolicy()
        ctx = _make_context()
        with pytest.raises(ValueError, match="No models available"):
            policy.route(ctx, [])

    def test_route_explores_uniformly(self) -> None:
        """Before min_pulls, all models should get tried."""
        policy = BanditRouterPolicy(min_pulls=5)
        models = ["model-a", "model-b", "model-c"]
        ctx = _make_context()

        seen = set()
        for _ in range(60):
            selected = policy.route(ctx, models)
            assert selected in models
            seen.add(selected)

        # With 60 random selections from 3 models, all should appear
        assert seen == set(models)

    def test_thompson_sampling(self) -> None:
        """After updates, higher-reward model should be selected more often."""
        policy = BanditRouterPolicy(strategy="thompson", min_pulls=2)
        models = ["good-model", "bad-model"]

        # Give good-model high rewards, bad-model low rewards
        for _ in range(20):
            policy.update("short", "good-model", 0.95)
            policy.update("short", "bad-model", 0.05)

        ctx = _make_context("hi", query_length=5)
        counts = {"good-model": 0, "bad-model": 0}
        for _ in range(100):
            selected = policy.route(ctx, models)
            counts[selected] += 1

        assert counts["good-model"] > 60, (
            f"good-model only selected {counts['good-model']}/100 times"
        )

    def test_ucb_selection(self) -> None:
        """After updates, UCB should favor the higher-reward model."""
        policy = BanditRouterPolicy(strategy="ucb", min_pulls=2)
        models = ["best-model", "worst-model"]

        # Train with clear reward difference
        for _ in range(20):
            policy.update("short", "best-model", 0.9)
            policy.update("short", "worst-model", 0.1)

        ctx = _make_context("hi", query_length=5)
        counts = {"best-model": 0, "worst-model": 0}
        for _ in range(100):
            selected = policy.route(ctx, models)
            counts[selected] += 1

        assert counts["best-model"] > 60, (
            f"best-model only selected {counts['best-model']}/100 times"
        )

    def test_update_increments_stats(self) -> None:
        policy = BanditRouterPolicy(reward_threshold=0.5)

        policy.update("code", "model-a", 0.8)  # success
        policy.update("code", "model-a", 0.3)  # failure
        policy.update("code", "model-a", 0.9)  # success

        stats = policy.get_stats("code")
        assert stats["model-a"]["pulls"] == 3
        assert stats["model-a"]["successes"] == 2
        assert stats["model-a"]["failures"] == 1
        assert abs(stats["model-a"]["mean_reward"] - (0.8 + 0.3 + 0.9) / 3) < 1e-9

    def test_get_stats(self) -> None:
        policy = BanditRouterPolicy()
        policy.update("code", "model-a", 0.8)
        policy.update("math", "model-b", 0.6)

        # Per-class stats
        code_stats = policy.get_stats("code")
        assert "model-a" in code_stats
        assert code_stats["model-a"]["pulls"] == 1

        # All stats
        all_stats = policy.get_stats()
        assert "code" in all_stats
        assert "math" in all_stats
        assert "model-a" in all_stats["code"]
        assert "model-b" in all_stats["math"]

    def test_get_stats_empty_class(self) -> None:
        policy = BanditRouterPolicy()
        stats = policy.get_stats("nonexistent")
        assert stats == {}

    def test_reset(self) -> None:
        policy = BanditRouterPolicy()
        policy.update("code", "model-a", 0.8)
        policy.update("math", "model-b", 0.6)
        assert policy._total_pulls == 2

        policy.reset()

        assert policy._total_pulls == 0
        assert policy.get_stats() == {}

    def test_registry_registration(self) -> None:
        ensure_registered()
        assert RouterPolicyRegistry.contains("bandit")
        assert RouterPolicyRegistry.get("bandit") is BanditRouterPolicy

    def test_under_explored_arms(self) -> None:
        """Models with fewer than min_pulls should be explored first."""
        policy = BanditRouterPolicy(strategy="thompson", min_pulls=5)
        models = ["explored", "unexplored"]

        # Give "explored" enough pulls
        for _ in range(10):
            policy.update("short", "explored", 0.9)

        ctx = _make_context("hi", query_length=5)
        # "unexplored" has 0 pulls < min_pulls=5, so it should always be chosen
        for _ in range(10):
            selected = policy.route(ctx, models)
            assert selected == "unexplored"

    def test_arm_stats_mean_reward_zero_pulls(self) -> None:
        stats = ArmStats()
        assert stats.mean_reward == 0.0

    def test_arm_stats_mean_reward(self) -> None:
        stats = ArmStats(total_reward=3.0, pulls=6)
        assert stats.mean_reward == 0.5

    def test_different_query_classes_independent(self) -> None:
        """Arms for different query classes should be independent."""
        policy = BanditRouterPolicy(min_pulls=1)

        # model-a is good for code, model-b is good for math
        for _ in range(20):
            policy.update("code", "model-a", 0.95)
            policy.update("code", "model-b", 0.1)
            policy.update("math", "model-b", 0.95)
            policy.update("math", "model-a", 0.1)

        code_stats = policy.get_stats("code")
        math_stats = policy.get_stats("math")

        assert (
            code_stats["model-a"]["mean_reward"]
            > code_stats["model-b"]["mean_reward"]
        )
        assert (
            math_stats["model-b"]["mean_reward"]
            > math_stats["model-a"]["mean_reward"]
        )
