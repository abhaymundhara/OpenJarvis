# Evals Framework

## Overview
The eval framework lives in `src/openjarvis/evals/` and is run via:
```bash
uv run python -m openjarvis.evals --config src/openjarvis/evals/configs/<config>.toml
```

## Structure
- **CLI**: `evals/cli.py` — command-line entry point
- **Core**: `evals/core/` — config loading (`config.py`), types (`types.py`), runner
- **Configs**: `evals/configs/` — TOML config files defining eval runs
- **Datasets**: `evals/datasets/` — dataset loaders (one file per benchmark)
- **Scorers**: `evals/scorers/` — scoring logic (one file per benchmark)

## Available Datasets
SuperGPQA, GPQA, MMLU-Pro, MATH-500, GAIA, SWE-bench, FRAMES, SimpleQA, TerminalBench, PaperArena, LifelongAgent, and more.

## Config Format
Eval configs are TOML files that specify:
- Model / engine to evaluate
- Dataset(s) to run
- Scoring method
- Number of samples, seeds, and other parameters

## Lint
E501 (line length) is relaxed for `evals/datasets/*.py` and `evals/scorers/*.py`.
