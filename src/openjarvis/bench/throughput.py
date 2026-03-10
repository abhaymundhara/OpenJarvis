"""Throughput benchmark — measures tokens per second."""

from __future__ import annotations

import logging
import time
from typing import Any

from openjarvis.bench._stubs import BaseBenchmark, BenchmarkResult
from openjarvis.core.registry import BenchmarkRegistry
from openjarvis.core.types import Message, Role
from openjarvis.engine._stubs import InferenceEngine

logger = logging.getLogger(__name__)


class ThroughputBenchmark(BaseBenchmark):
    """Measures inference throughput in tokens per second."""

    @property
    def name(self) -> str:
        return "throughput"

    @property
    def description(self) -> str:
        return "Measures inference throughput in tokens per second"

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
        prompt = "Write a short paragraph about artificial intelligence."
        messages = [Message(role=Role.USER, content=prompt)]

        for _ in range(warmup_samples):
            try:
                engine.generate(messages, model=model)
            except Exception as exc:
                logger.debug("Warmup request failed: %s", exc)

        total_tokens = 0
        total_time = 0.0
        errors = 0

        for _ in range(num_samples):
            t0 = time.time()
            try:
                result = engine.generate(messages, model=model)
                elapsed = time.time() - t0
                usage = result.get("usage", {})
                tokens = usage.get("completion_tokens", 0)
                total_tokens += tokens
                total_time += elapsed
            except Exception as exc:
                logger.debug("Measurement request failed: %s", exc)
                errors += 1

        tps = total_tokens / total_time if total_time > 0 else 0.0

        return BenchmarkResult(
            benchmark_name=self.name,
            model=model,
            engine=engine.engine_id,
            metrics={
                "tokens_per_second": tps,
                "total_tokens": float(total_tokens),
                "total_time_seconds": total_time,
            },
            samples=num_samples,
            errors=errors,
        )


def ensure_registered() -> None:
    """Register the throughput benchmark if not already present."""
    if not BenchmarkRegistry.contains("throughput"):
        BenchmarkRegistry.register_value("throughput", ThroughputBenchmark)


__all__ = ["ThroughputBenchmark"]
