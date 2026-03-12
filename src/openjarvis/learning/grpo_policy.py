"""GRPO router — Group Relative Policy Optimization for query→model routing."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List

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
class GRPOSample:
    """A single sample in a GRPO group."""
    query_class: str
    model: str
    reward: float


@dataclass
class GRPOState:
    """Persistent state for GRPO policy weights."""
    # model -> query_class -> weight (log probability)
    weights: Dict[str, Dict[str, float]] = field(
        default_factory=lambda: defaultdict(
            lambda: defaultdict(float)
        ),
    )
    # Track sample counts for min_samples threshold
    sample_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    total_updates: int = 0


class GRPORouterPolicy:
    """Group Relative Policy Optimization for routing queries to models.

    Groups samples by query_class, computes relative advantage within each
    group (reward - mean_reward) / std, and updates policy weights via
    softmax gradient.

    Falls back to random selection when insufficient samples exist.
    """

    def __init__(
        self,
        *,
        learning_rate: float = 0.1,
        min_samples: int = 5,
        group_size: int = 4,
        temperature: float = 1.0,
    ) -> None:
        self._lr = learning_rate
        self._min_samples = min_samples
        self._group_size = group_size
        self._temperature = temperature
        self._state = GRPOState()
        self._sample_buffer: List[GRPOSample] = []

    def route(self, context: RoutingContext, models: List[str]) -> str:
        """Select the best model for the given routing context."""
        if not models:
            raise ValueError("No models available for routing")

        query_class = _derive_query_class(context)

        # Check if we have enough samples
        if self._state.sample_counts.get(query_class, 0) < self._min_samples:
            return random.choice(models)

        # Compute softmax probabilities from weights
        scores = []
        for m in models:
            w = self._state.weights.get(m, {}).get(query_class, 0.0)
            scores.append(w / self._temperature)

        # Softmax
        max_score = max(scores)
        exp_scores = [math.exp(s - max_score) for s in scores]
        total = sum(exp_scores)
        probs = [e / total for e in exp_scores]

        # Sample from distribution
        r = random.random()
        cumulative = 0.0
        for i, p in enumerate(probs):
            cumulative += p
            if r <= cumulative:
                return models[i]
        return models[-1]

    def add_sample(self, query_class: str, model: str, reward: float) -> None:
        """Add a training sample to the buffer."""
        self._sample_buffer.append(GRPOSample(
            query_class=query_class, model=model, reward=reward,
        ))
        self._state.sample_counts[query_class] = (
            self._state.sample_counts.get(query_class, 0) + 1
        )

    def update(self) -> Dict[str, Any]:
        """Run GRPO update on accumulated samples.

        Groups samples by query_class, computes relative advantages,
        and updates policy weights.

        Returns stats about the update.
        """
        if not self._sample_buffer:
            return {"updated": False, "reason": "no samples"}

        # Group by query_class
        groups: Dict[str, List[GRPOSample]] = defaultdict(list)
        for sample in self._sample_buffer:
            groups[sample.query_class].append(sample)

        updates_applied = 0
        for qc, samples in groups.items():
            if len(samples) < 2:
                continue  # Need at least 2 for relative comparison

            # Compute group statistics
            rewards = [s.reward for s in samples]
            mean_r = sum(rewards) / len(rewards)
            var_r = sum((r - mean_r) ** 2 for r in rewards) / len(rewards)
            std_r = math.sqrt(var_r) if var_r > 0 else 1.0

            # Compute advantages and update weights
            for sample in samples:
                advantage = (sample.reward - mean_r) / std_r
                self._state.weights[sample.model][qc] += (
                    self._lr * advantage
                )
                updates_applied += 1

        self._state.total_updates += 1
        processed = len(self._sample_buffer)
        self._sample_buffer.clear()

        return {
            "updated": True,
            "samples_processed": processed,
            "groups": len(groups),
            "updates_applied": updates_applied,
            "total_updates": self._state.total_updates,
        }

    @property
    def state(self) -> GRPOState:
        """Access the current policy state."""
        return self._state

    def reset(self) -> None:
        """Reset all state."""
        self._state = GRPOState()
        self._sample_buffer.clear()


def ensure_registered() -> None:
    """Register GRPORouterPolicy if not already present."""
    if not RouterPolicyRegistry.contains("grpo"):
        RouterPolicyRegistry.register_value("grpo", GRPORouterPolicy)


ensure_registered()

__all__ = ["GRPORouterPolicy", "GRPOSample", "GRPOState"]
