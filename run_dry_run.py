#!/usr/bin/env python3
"""Run the multi-benchmark optimization dry run directly (bypasses CLI)."""

import logging
import os
import sys

# Load env
from pathlib import Path

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
log = logging.getLogger("dry_run")

from openjarvis.optimize.config import load_benchmark_specs, load_objectives, load_optimize_config
from openjarvis.optimize.llm_optimizer import LLMOptimizer
from openjarvis.optimize.optimizer import OptimizationEngine
from openjarvis.optimize.search_space import build_search_space
from openjarvis.optimize.store import OptimizationStore
from openjarvis.optimize.trial_runner import MultiBenchTrialRunner

CONFIG = "src/openjarvis/optimize/configs/qwen3-235b-dry-run.toml"

log.info("Loading config: %s", CONFIG)
data = load_optimize_config(CONFIG)
opt = data["optimize"]

search_space = build_search_space(data)
objectives = load_objectives(data)
specs = load_benchmark_specs(data)

log.info("Benchmarks: %s", [(s.benchmark, s.max_samples, s.weight) for s in specs])
log.info("Objectives: %s", [(o.metric, o.direction, o.weight) for o in objectives])

optimizer_model = opt.get("optimizer_model", "claude-opus-4-6")
output_dir = opt.get("output_dir", "results/optimize/dry-run/")
max_trials = opt.get("max_trials", 2)
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

log.info("Starting dry run: max_trials=%d", max_trials)

run = engine.run(
    progress_callback=lambda t, m: log.info("Trial %d/%d complete", t, m),
)

store.close()

log.info("Optimization complete!")
log.info("  Run ID:   %s", run.run_id)
log.info("  Status:   %s", run.status)
log.info("  Trials:   %d", len(run.trials))
if run.best_trial:
    log.info(
        "  Best: %s (accuracy=%.4f)",
        run.best_trial.trial_id,
        run.best_trial.accuracy,
    )

# Export best recipe
if run.best_trial:
    recipe_path = Path(output_dir) / "best_recipe.toml"
    engine.export_best_recipe(run, recipe_path)
    log.info("  Best recipe exported to: %s", recipe_path)
