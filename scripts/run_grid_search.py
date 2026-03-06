#!/usr/bin/env python3
"""Grid-search runner: models x engines x agents x benchmarks.

Iterates all valid (model, engine, agent, benchmark) combinations,
running each with ``max_samples=5`` by default.  Supports ``--dry-run``
to preview the matrix and ``--resume`` to skip completed runs.

Usage::

    # Preview the full matrix
    uv run python scripts/run_grid_search.py --dry-run

    # Run everything (5 samples each)
    uv run python scripts/run_grid_search.py

    # Resume after partial completion
    uv run python scripts/run_grid_search.py --resume

    # Custom sample count
    uv run python scripts/run_grid_search.py -n 10

    # Filter to specific model / engine / agent / benchmark
    uv run python scripts/run_grid_search.py --model "openai/gpt-oss-120b" --engine vllm
    uv run python scripts/run_grid_search.py --agent simple --benchmark supergpqa
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

# ---------------------------------------------------------------------------
# Grid dimensions
# ---------------------------------------------------------------------------

MODELS: Dict[str, List[str]] = {
    "openai/gpt-oss-120b": ["vllm", "sglang"],
    "Qwen/Qwen3.5-122B-A10B-FP8": ["vllm", "sglang"],
    "unsloth/Qwen3.5-397B-A17B-GGUF": ["llamacpp", "ollama"],
    "unsloth/Kimi-K2.5-GGUF": ["llamacpp", "ollama"],
    "unsloth/GLM-5-GGUF": ["llamacpp", "ollama"],
}

AGENTS = ["simple", "orchestrator", "native_react", "native_openhands", "rlm"]

TOOL_SETS: Dict[str, List[str]] = {
    "simple": [],
    "orchestrator": ["calculator", "think", "code_interpreter"],
    "native_react": ["calculator", "think", "code_interpreter"],
    "native_openhands": [
        "code_interpreter", "web_search", "file_read", "calculator", "think",
    ],
    "rlm": ["calculator", "think", "code_interpreter"],
}

BENCHMARKS = [
    "supergpqa", "gpqa", "mmlu-pro", "math500", "natural-reasoning",
    "hle", "simpleqa", "wildchat", "ipw", "gaia", "frames",
    "swebench", "swefficiency", "terminalbench", "terminalbench-native",
]

LOG = logging.getLogger("grid-search")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class RunSpec:
    """A single cell in the grid."""

    model: str
    engine: str
    agent: str
    benchmark: str
    tools: List[str]

    @property
    def slug(self) -> str:
        model_slug = self.model.replace("/", "-").replace(":", "-")
        return f"{model_slug}/{self.engine}/{self.agent}/{self.benchmark}"

    def output_path(self, base_dir: Path) -> Path:
        model_slug = self.model.replace("/", "-").replace(":", "-")
        return base_dir / model_slug / self.engine / self.agent / f"{self.benchmark}.jsonl"

    def summary_path(self, base_dir: Path) -> Path:
        return self.output_path(base_dir).with_suffix(".summary.json")


def build_matrix(
    *,
    model_filter: Optional[str] = None,
    engine_filter: Optional[str] = None,
    agent_filter: Optional[str] = None,
    benchmark_filter: Optional[str] = None,
) -> List[RunSpec]:
    """Enumerate all valid (model, engine, agent, benchmark) combinations."""
    specs: List[RunSpec] = []
    for model, engines in MODELS.items():
        if model_filter and model_filter not in model:
            continue
        for engine in engines:
            if engine_filter and engine != engine_filter:
                continue
            for agent in AGENTS:
                if agent_filter and agent != agent_filter:
                    continue
                tools = list(TOOL_SETS[agent])
                for bench in BENCHMARKS:
                    if benchmark_filter and benchmark_filter not in bench:
                        continue
                    specs.append(RunSpec(
                        model=model,
                        engine=engine,
                        agent=agent,
                        benchmark=bench,
                        tools=tools,
                    ))
    return specs


def is_completed(spec: RunSpec, base_dir: Path) -> bool:
    """Check whether a run already has a summary file on disk."""
    return spec.summary_path(base_dir).exists()


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


def run_single(spec: RunSpec, *, max_samples: int, base_dir: Path) -> Optional[dict]:
    """Execute a single eval run and return the summary dict (or None on error)."""
    from openjarvis.evals.core.types import RunConfig

    output = spec.output_path(base_dir)
    output.parent.mkdir(parents=True, exist_ok=True)

    backend = "jarvis-agent" if spec.agent != "simple" else "jarvis-direct"

    config = RunConfig(
        benchmark=spec.benchmark,
        backend=backend,
        model=spec.model,
        max_samples=max_samples,
        max_workers=1,
        temperature=0.0,
        max_tokens=2048,
        judge_model="gpt-5-mini-2025-08-07",
        engine_key=spec.engine,
        agent_name=spec.agent if spec.agent != "simple" else None,
        tools=spec.tools,
        output_path=str(output),
        seed=42,
    )

    from openjarvis.evals.cli import _run_single

    try:
        summary = _run_single(config)
        summary_dict = {
            "model": spec.model,
            "engine": spec.engine,
            "agent": spec.agent,
            "benchmark": spec.benchmark,
            "accuracy": getattr(summary, "accuracy", 0.0),
            "correct": getattr(summary, "correct", 0),
            "scored_samples": getattr(summary, "scored_samples", 0),
            "total_samples": getattr(summary, "total_samples", 0),
        }
        spec.summary_path(base_dir).write_text(
            json.dumps(summary_dict, indent=2) + "\n",
        )
        return summary_dict
    except Exception as exc:
        LOG.error("FAILED %s: %s", spec.slug, exc)
        error_dict = {
            "model": spec.model,
            "engine": spec.engine,
            "agent": spec.agent,
            "benchmark": spec.benchmark,
            "error": str(exc),
        }
        spec.summary_path(base_dir).write_text(
            json.dumps(error_dict, indent=2) + "\n",
        )
        return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run eval grid search: models x engines x agents x benchmarks",
    )
    parser.add_argument(
        "-n", "--max-samples", type=int, default=5,
        help="Max samples per benchmark (default: 5)",
    )
    parser.add_argument(
        "-o", "--output-dir", default="results/grid-search",
        help="Base output directory (default: results/grid-search)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the matrix without running anything",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip runs that already have summary files on disk",
    )
    parser.add_argument("--model", default=None, help="Filter to model (substring match)")
    parser.add_argument("--engine", default=None, help="Filter to engine (exact match)")
    parser.add_argument("--agent", default=None, help="Filter to agent (exact match)")
    parser.add_argument("--benchmark", default=None, help="Filter to benchmark (substring match)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    base_dir = Path(args.output_dir)
    matrix = build_matrix(
        model_filter=args.model,
        engine_filter=args.engine,
        agent_filter=args.agent,
        benchmark_filter=args.benchmark,
    )

    # Count dimensions for display
    models_in_grid = sorted({s.model for s in matrix})
    engines_in_grid = sorted({s.engine for s in matrix})
    agents_in_grid = sorted({s.agent for s in matrix})
    benches_in_grid = sorted({s.benchmark for s in matrix})

    print("=" * 72)
    print("OpenJarvis Eval Grid Search")
    print("=" * 72)
    print(f"  Models:     {len(models_in_grid):>4}  {', '.join(models_in_grid)}")
    print(f"  Engines:    {len(engines_in_grid):>4}  {', '.join(engines_in_grid)}")
    print(f"  Agents:     {len(agents_in_grid):>4}  {', '.join(agents_in_grid)}")
    print(f"  Benchmarks: {len(benches_in_grid):>4}  {', '.join(benches_in_grid)}")
    print(f"  Total runs: {len(matrix):>4}")
    print(f"  Samples:    {args.max_samples:>4} per run")
    print(f"  Output:     {base_dir}")
    print("=" * 72)

    if args.dry_run:
        print("\n[DRY RUN] Full matrix:\n")
        for i, spec in enumerate(matrix, 1):
            status = ""
            if args.resume and is_completed(spec, base_dir):
                status = " [SKIP]"
            print(f"  {i:>4}. {spec.slug}{status}")
        skipped = sum(1 for s in matrix if is_completed(s, base_dir))
        print(f"\n  Total: {len(matrix)} | Already completed: {skipped} | Remaining: {len(matrix) - skipped}")
        return

    # Actual execution
    results: List[dict] = []
    skipped = 0
    failed = 0
    t0 = time.monotonic()

    for i, spec in enumerate(matrix, 1):
        if args.resume and is_completed(spec, base_dir):
            LOG.info("[%d/%d] SKIP (completed) %s", i, len(matrix), spec.slug)
            skipped += 1
            continue

        LOG.info("[%d/%d] RUNNING %s", i, len(matrix), spec.slug)
        result = run_single(spec, max_samples=args.max_samples, base_dir=base_dir)
        if result is not None:
            results.append(result)
            LOG.info(
                "  -> accuracy=%.4f (%d/%d)",
                result["accuracy"], result["correct"], result["scored_samples"],
            )
        else:
            failed += 1

    elapsed = time.monotonic() - t0
    print("\n" + "=" * 72)
    print("Grid Search Complete")
    print("=" * 72)
    print(f"  Ran:     {len(results)}")
    print(f"  Skipped: {skipped}")
    print(f"  Failed:  {failed}")
    print(f"  Time:    {elapsed:.1f}s")
    print(f"  Results: {base_dir}/")

    # Write consolidated results
    if results:
        consolidated = base_dir / "grid-results.jsonl"
        with open(consolidated, "a") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        print(f"  Summary: {consolidated}")


if __name__ == "__main__":
    main()
