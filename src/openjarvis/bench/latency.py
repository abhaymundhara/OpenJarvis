"""Latency benchmark — measures per-call inference latency."""

from __future__ import annotations

import logging
import statistics
import time
from typing import Any, List

from openjarvis.bench._stubs import BaseBenchmark, BenchmarkResult
from openjarvis.core.registry import BenchmarkRegistry
from openjarvis.core.types import Message, Role
from openjarvis.engine._stubs import InferenceEngine

logger = logging.getLogger(__name__)

_CANNED_PROMPTS = [
    "Hello",
    "What is 2+2?",
    "Explain gravity in one sentence",
]


class LatencyBenchmark(BaseBenchmark):
    """Measures per-call inference latency with short prompts."""

    @property
    def name(self) -> str:
        return "latency"

    @property
    def description(self) -> str:
        return "Measures per-call inference latency with short prompts"

    def run(
        self,
        engine: InferenceEngine,
        model: str,
        *,
        num_samples: int = 10,
        warmup_samples: int = 0,
        **kwargs: Any,
    ) -> BenchmarkResult:
        # Run warmup iterations (discarded)
        for i in range(warmup_samples):
            prompt = _CANNED_PROMPTS[i % len(_CANNED_PROMPTS)]
            messages = [Message(role=Role.USER, content=prompt)]
            try:
                engine.generate(messages, model=model)
            except Exception as exc:
                logger.debug("Warmup request failed: %s", exc)

        latencies: List[float] = []
        errors = 0

        for i in range(num_samples):
            prompt = _CANNED_PROMPTS[i % len(_CANNED_PROMPTS)]
            messages = [Message(role=Role.USER, content=prompt)]
            t0 = time.time()
            try:
                engine.generate(messages, model=model)
                latencies.append(time.time() - t0)
            except Exception as exc:
                logger.debug("Measurement request failed: %s", exc)
                errors += 1

        if not latencies:
            return BenchmarkResult(
                benchmark_name=self.name,
                model=model,
                engine=engine.engine_id,
                metrics={},
                samples=num_samples,
                errors=errors,
            )

        return BenchmarkResult(
            benchmark_name=self.name,
            model=model,
            engine=engine.engine_id,
            metrics={
                "mean_latency": statistics.mean(latencies),
                "p50_latency": statistics.median(latencies),
                "p95_latency": _percentile(latencies, 0.95),
                "min_latency": min(latencies),
                "max_latency": max(latencies),
                "std_latency": (
                    statistics.stdev(latencies) if len(latencies) > 1 else 0.0
                ),
            },
            samples=num_samples,
            errors=errors,
        )


def _percentile(data: List[float], p: float) -> float:
    """Compute the p-th percentile of a sorted list."""
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * p
    f = int(k)
    c = f + 1
    if c >= len(sorted_data):
        return sorted_data[-1]
    return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])


def ensure_registered() -> None:
    """Register the latency benchmark if not already present."""
    if not BenchmarkRegistry.contains("latency"):
        BenchmarkRegistry.register_value("latency", LatencyBenchmark)


__all__ = ["LatencyBenchmark"]
