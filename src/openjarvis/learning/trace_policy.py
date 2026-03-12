"""Trace-driven router policy — learns from interaction history.

Unlike the ``HeuristicRouter`` which uses static rules, this policy
learns from accumulated traces which model/agent/tool combinations
produce the best outcomes for different query types.  It maintains a
lightweight mapping of (query_class → model) that is updated
periodically from the ``TraceAnalyzer``.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from openjarvis.core.registry import RouterPolicyRegistry
from openjarvis.core.types import RoutingContext
from openjarvis.learning._stubs import RouterPolicy
from openjarvis.traces.analyzer import TraceAnalyzer

# Query classification for grouping traces
_CODE_RE = re.compile(
    r"```|`[^`]+`|\bdef\s|\bclass\s|\bimport\s|\bfunction\s",
    re.IGNORECASE,
)
_MATH_RE = re.compile(
    r"\bsolve\b|\bintegral\b|\bequation\b|\bcalculate\b|\bcompute\b",
    re.IGNORECASE,
)


def classify_query(query: str) -> str:
    """Classify a query into a broad category for routing."""
    if _CODE_RE.search(query):
        return "code"
    if _MATH_RE.search(query):
        return "math"
    if len(query) < 50:
        return "short"
    if len(query) > 500:
        return "long"
    return "general"


class TraceDrivenPolicy(RouterPolicy):
    """Router policy that learns from historical traces.

    Maintains a mapping of ``query_class → best_model`` derived from
    trace outcomes.  Falls back to the provided default when no trace
    data is available for a query class.

    The policy is updated by calling :meth:`update_from_traces`, which
    reads the ``TraceAnalyzer`` and recomputes the mapping.
    """

    def __init__(
        self,
        analyzer: Optional[TraceAnalyzer] = None,
        *,
        available_models: Optional[List[str]] = None,
        default_model: str = "",
        fallback_model: str = "",
    ) -> None:
        self._analyzer = analyzer
        self._available = available_models or []
        self._default = default_model
        self._fallback = fallback_model
        # Learned mapping: query_class → model key
        self._policy_map: Dict[str, str] = {}
        # Track confidence: query_class → sample count
        self._confidence: Dict[str, int] = {}
        # Minimum samples before trusting learned policy
        self.min_samples: int = 5

    @property
    def policy_map(self) -> Dict[str, str]:
        """Current learned routing decisions (read-only copy)."""
        return dict(self._policy_map)

    def select_model(self, context: RoutingContext) -> str:
        """Select the best model based on learned policy or fallback."""
        query_class = classify_query(context.query)

        # Use learned policy if we have enough confidence
        if (
            query_class in self._policy_map
            and self._confidence.get(query_class, 0) >= self.min_samples
        ):
            model = self._policy_map[query_class]
            if not self._available or model in self._available:
                return model

        # Fallback chain
        avail = self._available
        if self._default and (not avail or self._default in avail):
            return self._default
        if self._fallback and (not avail or self._fallback in avail):
            return self._fallback
        if self._available:
            return self._available[0]
        return self._default or ""

    def update_from_traces(
        self,
        *,
        since: Optional[float] = None,
        until: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Recompute the policy map from trace history.

        Returns a summary of what changed for logging/debugging.
        """
        if self._analyzer is None:
            return {"error": "no analyzer configured"}

        traces = self._analyzer._store.list_traces(
            since=since, until=until, limit=10_000
        )
        if not traces:
            return {"updated": False, "reason": "no traces"}

        # Group traces by query class
        groups: Dict[str, list] = {}
        for t in traces:
            qclass = classify_query(t.query)
            groups.setdefault(qclass, []).append(t)

        old_map = dict(self._policy_map)
        changes: Dict[str, Dict[str, str]] = {}

        for qclass, class_traces in groups.items():
            # Score each model for this query class
            model_scores: Dict[str, _ModelScore] = {}
            for t in class_traces:
                if not t.model:
                    continue
                if t.model not in model_scores:
                    model_scores[t.model] = _ModelScore()
                score = model_scores[t.model]
                score.count += 1
                score.total_latency += t.total_latency_seconds
                if t.outcome == "success":
                    score.successes += 1
                if t.feedback is not None:
                    score.feedback_sum += t.feedback
                    score.feedback_count += 1

            if not model_scores:
                continue

            # Pick the best model: weighted score of success_rate and feedback
            best_model = max(
                model_scores.items(),
                key=lambda kv: kv[1].composite_score(),
            )[0]

            self._policy_map[qclass] = best_model
            self._confidence[qclass] = sum(s.count for s in model_scores.values())

            if old_map.get(qclass) != best_model:
                changes[qclass] = {
                    "old": old_map.get(qclass, ""),
                    "new": best_model,
                }

        return {
            "updated": True,
            "query_classes": len(groups),
            "total_traces": len(traces),
            "changes": changes,
        }

    def observe(
        self,
        query: str,
        model: str,
        outcome: Optional[str],
        feedback: Optional[float],
    ) -> None:
        """Record a single observation for online (incremental) updates.

        This is a lighter-weight alternative to :meth:`update_from_traces`
        for use cases where you want to update the policy after every
        interaction rather than in batch.
        """
        qclass = classify_query(query)
        current_count = self._confidence.get(qclass, 0)

        # Simple exponential moving average for online update
        if qclass not in self._policy_map:
            self._policy_map[qclass] = model
            self._confidence[qclass] = 1
            return

        self._confidence[qclass] = current_count + 1

        # Only switch models if the new model shows clearly better outcomes
        if outcome == "success" and feedback is not None and feedback > 0.7:
            # Weight new evidence against existing policy
            if current_count < self.min_samples:
                self._policy_map[qclass] = model


class _ModelScore:
    """Accumulator for per-model scoring during policy update."""

    __slots__ = (
        "count", "successes", "total_latency",
        "feedback_sum", "feedback_count",
    )

    def __init__(self) -> None:
        self.count = 0
        self.successes = 0
        self.total_latency = 0.0
        self.feedback_sum = 0.0
        self.feedback_count = 0

    def composite_score(self) -> float:
        """Weighted score combining success rate and feedback."""
        sr = self.successes / self.count if self.count else 0.0
        fb = (
            self.feedback_sum / self.feedback_count
            if self.feedback_count else 0.5
        )
        # 60% success rate + 40% feedback
        return 0.6 * sr + 0.4 * fb


def ensure_registered() -> None:
    """Register TraceDrivenPolicy if not already present."""
    if not RouterPolicyRegistry.contains("learned"):
        RouterPolicyRegistry.register_value("learned", TraceDrivenPolicy)


ensure_registered()


__all__ = ["TraceDrivenPolicy", "classify_query", "ensure_registered"]
