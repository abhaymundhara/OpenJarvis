#!/usr/bin/env python3
"""Run the full multi-benchmark optimization (15 trials, 200 samples/benchmark)."""

import logging
import os
import sys

from pathlib import Path

# Load env
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            if line.startswith("export "):
                line = line[7:]
            key, _, val = line.partition("=")
            os.environ[key] = val

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
log = logging.getLogger("full_optimization")

from openjarvis.optimize.config import load_benchmark_specs, load_objectives, load_optimize_config
from openjarvis.optimize.llm_optimizer import LLMOptimizer
from openjarvis.optimize.optimizer import OptimizationEngine
from openjarvis.optimize.search_space import build_search_space
from openjarvis.optimize.store import OptimizationStore
from openjarvis.optimize.trial_runner import MultiBenchTrialRunner

CONFIG = "src/openjarvis/optimize/configs/qwen3.5-joint-agentic.toml"

# Ensure HF_TOKEN is set for gated datasets (HLE)
if not os.environ.get("HF_TOKEN"):
    log.warning("HF_TOKEN not set — gated datasets like HLE may fail")

log.info("Loading config: %s", CONFIG)
data = load_optimize_config(CONFIG)
opt = data["optimize"]

search_space = build_search_space(data)
objectives = load_objectives(data)
specs = load_benchmark_specs(data)

log.info("Benchmarks: %s", [(s.benchmark, s.max_samples, s.weight) for s in specs])
log.info("Objectives: %s", [(o.metric, o.direction, o.weight) for o in objectives])

optimizer_model = opt.get("optimizer_model", "claude-opus-4-6")
output_dir = opt.get("output_dir", "results/optimize/qwen3-235b-joint/")
max_trials = opt.get("max_trials", 15)
early_stop = opt.get("early_stop_patience", 5)

# Build optimizer backend (Anthropic)
from openjarvis.evals.cli import _build_judge_backend

optimizer_backend = _build_judge_backend(optimizer_model)
log.info("Optimizer backend ready: %s", optimizer_model)

llm_opt = LLMOptimizer(
    search_space=search_space,
    optimizer_model=optimizer_model,
    optimizer_backend=optimizer_backend,
)

runner = MultiBenchTrialRunner(
    benchmark_specs=specs,
    output_dir=output_dir,
)

Path(output_dir).mkdir(parents=True, exist_ok=True)
store = OptimizationStore(Path(output_dir) / "optimize.db")

engine = OptimizationEngine(
    search_space=search_space,
    llm_optimizer=llm_opt,
    trial_runner=runner,
    store=store,
    max_trials=max_trials,
    early_stop_patience=early_stop,
)

log.info("Starting full optimization: max_trials=%d, benchmarks=%s",
         max_trials, [s.benchmark for s in specs])

run = engine.run(
    progress_callback=lambda t, m: log.info("=== Trial %d/%d complete ===", t, m),
)

store.close()

log.info("=" * 60)
log.info("OPTIMIZATION COMPLETE")
log.info("=" * 60)
log.info("  Run ID:   %s", run.run_id)
log.info("  Status:   %s", run.status)
log.info("  Trials:   %d", len(run.trials))
if run.best_trial:
    log.info(
        "  Best: %s (accuracy=%.4f)",
        run.best_trial.trial_id,
        run.best_trial.accuracy,
    )
    for t in run.trials:
        log.info(
            "    Trial %s: acc=%.4f lat=%.2fs",
            t.trial_id[:8], t.accuracy, t.mean_latency_seconds,
        )

# Export best recipe
if run.best_trial:
    recipe_path = Path(output_dir) / "best_recipe.toml"
    engine.export_best_recipe(run, recipe_path)
    log.info("  Best recipe exported to: %s", recipe_path)

# Pareto frontier
if run.pareto_frontier:
    log.info("  Pareto frontier: %d trials", len(run.pareto_frontier))
    for t in run.pareto_frontier:
        log.info("    %s: acc=%.4f lat=%.2fs energy=%.1fJ",
                 t.trial_id[:8], t.accuracy, t.mean_latency_seconds,
                 t.total_energy_joules)
