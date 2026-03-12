"""Bandit router — Thompson Sampling / UCB for query→model selection."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional

from openjarvis.core.registry import RouterPolicyRegistry
from openjarvis.core.types import RoutingContext


def _derive_query_class(context: RoutingContext) -> str:
    """Derive a query class string from RoutingContext fields."""
    if context.has_code:
        return "code"
    if context.has_math:
        return "math"
    if context.query_length < 50:
        return "short"
    if context.query_length > 500:
        return "long"
    return "general"


@dataclass(slots=True)
class ArmStats:
    """Statistics for a single arm (model)."""
    successes: int = 0  # alpha for Beta distribution
    failures: int = 0   # beta for Beta distribution
    total_reward: float = 0.0
    pulls: int = 0

    @property
    def mean_reward(self) -> float:
        return self.total_reward / self.pulls if self.pulls > 0 else 0.0


class BanditRouterPolicy:
    """Multi-armed bandit router using Thompson Sampling or UCB.

    Each (query_class, model) pair is an arm. Rewards come from
    trace outcomes.
    """

    def __init__(
        self,
        *,
        strategy: Literal["thompson", "ucb"] = "thompson",
        exploration_factor: float = 2.0,  # UCB exploration constant
        min_pulls: int = 3,  # minimum pulls before trusting estimates
        reward_threshold: float = 0.5,  # reward above this = success
    ) -> None:
        self._strategy = strategy
        self._exploration = exploration_factor
        self._min_pulls = min_pulls
        self._reward_threshold = reward_threshold
        # query_class -> model -> ArmStats
        self._arms: Dict[str, Dict[str, ArmStats]] = defaultdict(
            lambda: defaultdict(ArmStats)
        )
        self._total_pulls = 0

    def route(self, context: RoutingContext, models: List[str]) -> str:
        """Select model using the configured bandit strategy."""
        if not models:
            raise ValueError("No models available")

        query_class = _derive_query_class(context)
        arms = self._arms[query_class]

        # Ensure all models have arms
        for m in models:
            if m not in arms:
                arms[m] = ArmStats()

        # Check minimum pulls — explore uniformly first
        under_explored = [m for m in models if arms[m].pulls < self._min_pulls]
        if under_explored:
            return random.choice(under_explored)

        if self._strategy == "thompson":
            return self._thompson_select(models, arms)
        else:
            return self._ucb_select(models, arms)

    def _thompson_select(self, models: List[str], arms: Dict[str, ArmStats]) -> str:
        """Thompson Sampling: sample from Beta(alpha, beta) per arm."""
        best_model = models[0]
        best_sample = -1.0

        for m in models:
            stats = arms[m]
            alpha = stats.successes + 1  # Prior: Beta(1,1)
            beta = stats.failures + 1
            sample = random.betavariate(alpha, beta)
            if sample > best_sample:
                best_sample = sample
                best_model = m

        return best_model

    def _ucb_select(self, models: List[str], arms: Dict[str, ArmStats]) -> str:
        """UCB1: select arm with highest upper confidence bound."""
        best_model = models[0]
        best_ucb = -1.0

        total = max(self._total_pulls, 1)
        for m in models:
            stats = arms[m]
            if stats.pulls == 0:
                return m  # Unexplored arm gets priority

            mean = stats.mean_reward
            exploration = self._exploration * math.sqrt(
                math.log(total) / stats.pulls
            )
            ucb_value = mean + exploration

            if ucb_value > best_ucb:
                best_ucb = ucb_value
                best_model = m

        return best_model

    def update(self, query_class: str, model: str, reward: float) -> None:
        """Update arm statistics with observed reward."""
        stats = self._arms[query_class][model]
        stats.pulls += 1
        stats.total_reward += reward
        if reward >= self._reward_threshold:
            stats.successes += 1
        else:
            stats.failures += 1
        self._total_pulls += 1

    def get_stats(self, query_class: Optional[str] = None) -> Dict[str, Any]:
        """Get arm statistics."""
        if query_class:
            arms = self._arms.get(query_class, {})
            return {
                m: {
                    "pulls": s.pulls,
                    "mean_reward": s.mean_reward,
                    "successes": s.successes,
                    "failures": s.failures,
                }
                for m, s in arms.items()
            }
        return {
            qc: {
                m: {"pulls": s.pulls, "mean_reward": s.mean_reward}
                for m, s in arms.items()
            }
            for qc, arms in self._arms.items()
        }

    def reset(self) -> None:
        """Reset all state."""
        self._arms.clear()
        self._total_pulls = 0


def ensure_registered() -> None:
    """Register BanditRouterPolicy if not already present."""
    if not RouterPolicyRegistry.contains("bandit"):
        RouterPolicyRegistry.register_value("bandit", BanditRouterPolicy)


ensure_registered()

__all__ = ["ArmStats", "BanditRouterPolicy"]
