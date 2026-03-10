"""Energy benchmark — measures energy per token at thermal equilibrium."""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

from openjarvis.bench._stubs import BaseBenchmark, BenchmarkResult
from openjarvis.core.registry import BenchmarkRegistry
from openjarvis.core.types import Message, Role
from openjarvis.engine._stubs import InferenceEngine

logger = logging.getLogger(__name__)

_PROMPT = "Write a short paragraph about artificial intelligence."


class EnergyBenchmark(BaseBenchmark):
    """Measures energy per token at thermal equilibrium."""

    @property
    def name(self) -> str:
        return "energy"

    @property
    def description(self) -> str:
        return "Measures energy per token at thermal equilibrium"

    def run(
        self,
        engine: InferenceEngine,
        model: str,
        *,
        num_samples: int = 10,
        warmup_samples: int = 5,
        energy_monitor: Optional[Any] = None,
        **kwargs: Any,
    ) -> BenchmarkResult:
        messages = [Message(role=Role.USER, content=_PROMPT)]

        # --- Warmup phase (discarded) ---
        for _ in range(warmup_samples):
            try:
                engine.generate(messages, model=model)
            except Exception as exc:
                logger.debug("Warmup request failed: %s", exc)

        # --- Measurement phase ---
        total_tokens = 0
        total_energy = 0.0
        total_time = 0.0
        errors = 0
        energy_method = ""

        if energy_monitor is not None:
            from openjarvis.telemetry.batch import EnergyBatch
            from openjarvis.telemetry.steady_state import SteadyStateDetector

            detector = SteadyStateDetector()
            batch = EnergyBatch(energy_monitor=energy_monitor)

            with batch.sample() as ctx:
                for _ in range(num_samples):
                    t0 = time.time()
                    try:
                        result = engine.generate(messages, model=model)
                        elapsed = time.time() - t0
                        usage = result.get("usage", {})
                        tokens = usage.get("completion_tokens", 0)
                        ctx.record_request(tokens=tokens)
                        total_tokens += tokens
                        total_time += elapsed
                        throughput = tokens / elapsed if elapsed > 0 else 0.0
                        detector.record(throughput)
                    except Exception as exc:
                        logger.debug("Measurement request failed: %s", exc)
                        errors += 1

            if batch.metrics is not None:
                total_energy = batch.metrics.total_energy_joules
            energy_method = getattr(energy_monitor, "energy_method", lambda: "")()
            ss_result = detector.result
        else:
            # No energy monitor — still measure throughput
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
            ss_result = None

        tps = total_tokens / total_time if total_time > 0 else 0.0
        energy_per_token = (
            total_energy / total_tokens if total_tokens > 0 else 0.0
        )
        mean_power = total_energy / total_time if total_time > 0 else 0.0

        metrics = {
            "tokens_per_second": tps,
            "total_energy_joules": total_energy,
            "energy_per_token_joules": energy_per_token,
            "mean_power_watts": mean_power,
            "total_tokens": float(total_tokens),
            "total_time_seconds": total_time,
        }

        return BenchmarkResult(
            benchmark_name=self.name,
            model=model,
            engine=engine.engine_id,
            metrics=metrics,
            samples=num_samples,
            errors=errors,
            warmup_samples=warmup_samples,
            steady_state_samples=ss_result.steady_state_samples if ss_result else 0,
            steady_state_reached=ss_result.steady_state_reached if ss_result else False,
            total_energy_joules=total_energy,
            energy_per_token_joules=energy_per_token,
            energy_method=energy_method,
        )


def ensure_registered() -> None:
    """Register the energy benchmark if not already present."""
    if not BenchmarkRegistry.contains("energy"):
        BenchmarkRegistry.register_value("energy", EnergyBenchmark)


__all__ = ["EnergyBenchmark"]
