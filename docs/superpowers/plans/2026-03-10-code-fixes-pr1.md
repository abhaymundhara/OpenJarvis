# Code Fixes PR 1 — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix `jarvis doctor` labels, add minimal config mode, fix "pillars" → "primitives" in pyproject.toml, and add logging to remaining ~19 silent exception blocks.

**Architecture:** Four independent tasks touching disjoint files. Each task can be committed independently. All changes are additive (logging) or cosmetic (labels, config template).

**Tech Stack:** Python 3.10+, Click CLI, logging stdlib, ruff linter, pytest

**Spec:** `docs/superpowers/specs/2026-03-10-code-fixes-pr1-design.md`

---

## Chunk 1: Doctor Labels + Minimal Config + Identity Fix

### Task 1: `jarvis doctor` — Descriptive Optional Package Labels

**Files:**
- Modify: `src/openjarvis/cli/doctor_cmd.py:213-229` (full block: list + loop + try/except)
- Test: `tests/cli/test_doctor_labels.py` (new file)

- [ ] **Step 1: Write failing test**

Create `tests/cli/test_doctor_labels.py`:

```python
"""Tests for ``jarvis doctor`` optional dependency labels."""

from __future__ import annotations

from unittest import mock

from click.testing import CliRunner

from openjarvis.cli import cli

_real_import = __builtins__.__import__


def _selective_import_blocker(*blocked: str):
    """Return an __import__ replacement that blocks specific packages."""
    def _import(name, *args, **kwargs):
        if name in blocked:
            raise ImportError(f"mocked: {name} not installed")
        return _real_import(name, *args, **kwargs)
    return _import


class TestDoctorOptionalLabels:
    def test_labels_show_description(self) -> None:
        """Doctor output uses descriptive labels, not raw package names."""
        runner = CliRunner()
        result = runner.invoke(cli, ["doctor"])
        # Should show descriptive labels
        assert "REST API server" in result.output
        assert "SFT/GRPO training" in result.output
        assert "NVIDIA energy monitoring" in result.output
        # Should NOT show old vague labels
        assert "torch (for learning)" not in result.output
        assert "pynvml (GPU monitoring)" not in result.output

    def test_labels_show_install_hint_on_missing(self) -> None:
        """When a package is missing, show install hint in status."""
        with mock.patch("builtins.__import__", side_effect=_selective_import_blocker("zeus")):
            runner = CliRunner()
            result = runner.invoke(cli, ["doctor"])
        assert "Not installed (openjarvis[energy-apple])" in result.output
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/cli/test_doctor_labels.py -v --tb=short`
Expected: FAIL — old labels still in output

- [ ] **Step 3: Implement the fix**

In `src/openjarvis/cli/doctor_cmd.py`, replace lines 213-228:

```python
    optional_packages = [
        ("fastapi", "openjarvis[server]", "REST API server"),
        ("torch", "pip install torch", "SFT/GRPO training"),
        ("pynvml", "openjarvis[energy-nvidia]", "NVIDIA energy monitoring"),
        ("amdsmi", "openjarvis[energy-amd]", "AMD energy monitoring"),
        ("colbert", "openjarvis[memory-colbert]", "ColBERT memory backend"),
        ("zeus", "openjarvis[energy-apple]", "Apple Silicon energy monitoring"),
    ]
    for pkg, install_hint, description in optional_packages:
        try:
            __import__(pkg)
            results.append(
                CheckResult(f"Optional: {description}", "ok", "Installed")
            )
        except Exception:
            results.append(
                CheckResult(
                    f"Optional: {description}",
                    "warn",
                    f"Not installed ({install_hint})",
                )
            )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/cli/test_doctor_labels.py -v --tb=short`
Expected: PASS

- [ ] **Step 5: Lint check**

Run: `uv run ruff check src/openjarvis/cli/doctor_cmd.py tests/cli/test_doctor_labels.py`
Expected: All checks passed

- [ ] **Step 6: Commit**

```bash
git add src/openjarvis/cli/doctor_cmd.py tests/cli/test_doctor_labels.py
git commit -m "fix: use descriptive labels in jarvis doctor optional packages"
```

---

### Task 2: Minimal Config Default with `--full` Flag

**Files:**
- Modify: `src/openjarvis/core/config.py:1109` (add function before `generate_default_toml`)
- Modify: `src/openjarvis/core/config.py:1374` (add to `__all__`)
- Modify: `src/openjarvis/cli/init_cmd.py:12-19,152-166` (import + use)
- Test: `tests/cli/test_init_guidance.py` (add tests)

- [ ] **Step 1: Write failing tests**

Append to `tests/cli/test_init_guidance.py`:

```python
class TestMinimalConfig:
    def test_init_generates_minimal_by_default(self, tmp_path: Path) -> None:
        """Default jarvis init produces a short config."""
        config_dir = tmp_path / ".openjarvis"
        config_path = config_dir / "config.toml"
        with (
            mock.patch(
                "openjarvis.cli.init_cmd.DEFAULT_CONFIG_DIR", config_dir
            ),
            mock.patch(
                "openjarvis.cli.init_cmd.DEFAULT_CONFIG_PATH", config_path
            ),
        ):
            result = CliRunner().invoke(cli, ["init"])
        assert result.exit_code == 0
        content = config_path.read_text()
        # Minimal config should be short
        lines = [ln for ln in content.splitlines() if ln.strip()]
        assert len(lines) <= 30
        # Should have the reference hint
        assert "jarvis init --full" in content

    def test_init_full_generates_verbose_config(self, tmp_path: Path) -> None:
        """jarvis init --full produces the full reference config."""
        config_dir = tmp_path / ".openjarvis"
        config_path = config_dir / "config.toml"
        with (
            mock.patch(
                "openjarvis.cli.init_cmd.DEFAULT_CONFIG_DIR", config_dir
            ),
            mock.patch(
                "openjarvis.cli.init_cmd.DEFAULT_CONFIG_PATH", config_path
            ),
        ):
            result = CliRunner().invoke(cli, ["init", "--full"])
        assert result.exit_code == 0
        content = config_path.read_text()
        # Full config should have many sections
        assert "[engine.ollama]" in content
        assert "[server]" in content
        assert "[security]" in content
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/cli/test_init_guidance.py::TestMinimalConfig -v --tb=short`
Expected: FAIL — `--full` flag not recognized, config still verbose by default

- [ ] **Step 3: Add `generate_minimal_toml()` to `core/config.py`**

Insert before `generate_default_toml` (line 1109):

```python
def generate_minimal_toml(hw: HardwareInfo) -> str:
    """Render a minimal TOML config with only essential settings."""
    engine = recommend_engine(hw)
    model = recommend_model(hw, engine)
    gpu_comment = ""
    if hw.gpu:
        mem_label = (
            "unified memory" if hw.gpu.vendor == "apple" else "VRAM"
        )
        gpu_comment = (
            f"\n# GPU: {hw.gpu.name}"
            f" ({hw.gpu.vram_gb} GB {mem_label})"
        )
    return f"""\
# OpenJarvis configuration
# Hardware: {hw.cpu_brand} ({hw.cpu_count} cores, {hw.ram_gb} GB RAM){gpu_comment}
# Full reference config: jarvis init --full

[engine]
default = "{engine}"

[intelligence]
default_model = "{model}"

[agent]
default_agent = "simple"

[tools]
enabled = ["code_interpreter", "web_search", "file_read", "shell_exec"]
"""
```

Add `"generate_minimal_toml"` to the `__all__` list (after `"generate_default_toml"` on line 1374).

- [ ] **Step 4: Update `cli/init_cmd.py`**

Update the import block (lines 12-19):
```python
from openjarvis.core.config import (
    DEFAULT_CONFIG_DIR,
    DEFAULT_CONFIG_PATH,
    detect_hardware,
    generate_default_toml,
    generate_minimal_toml,
    recommend_engine,
    recommend_model,
)
```

Update the `init` click command to accept `--full`. Find the `@click.command()` decorator for `init` and add:
```python
@click.option("--full", "full_config", is_flag=True,
              help="Generate full reference config with all sections")
```

Add `full_config` parameter to the function signature.

Replace the config generation logic (around line 152-155):
```python
    if config:
        toml_content = config.read_text()
    else:
        if full_config:
            toml_content = generate_default_toml(hw)
        else:
            toml_content = generate_minimal_toml(hw)
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/cli/test_init_guidance.py -v --tb=short`
Expected: ALL PASS (existing + new)

- [ ] **Step 6: Lint check**

Run: `uv run ruff check src/openjarvis/core/config.py src/openjarvis/cli/init_cmd.py tests/cli/test_init_guidance.py`
Expected: All checks passed

- [ ] **Step 7: Commit**

```bash
git add src/openjarvis/core/config.py src/openjarvis/cli/init_cmd.py tests/cli/test_init_guidance.py
git commit -m "feat: minimal config by default, add jarvis init --full for reference config"
```

---

### Task 3: Identity Consistency — "pillars" → "primitives"

**Files:**
- Modify: `pyproject.toml:8`

- [ ] **Step 1: Make the change**

In `pyproject.toml` line 8, change:
```
description = "OpenJarvis — modular AI assistant backend with composable intelligence pillars"
```
to:
```
description = "OpenJarvis — modular AI assistant backend with composable intelligence primitives"
```

- [ ] **Step 2: Verify**

Run: `grep -r "pillar" pyproject.toml`
Expected: No output (no matches)

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "fix: update pyproject.toml description from pillars to primitives"
```

---

## Chunk 2: Silent Exception Remediation

### Task 4: Add logging to `cli/serve.py`

**Files:**
- Modify: `src/openjarvis/cli/serve.py:1-5` (add import)
- Modify: `src/openjarvis/cli/serve.py:81-82,108-109,112-113,209-210` (add logging)

- [ ] **Step 1: Add logger setup**

After `import sys` (line 5), add:
```python
import logging

logger = logging.getLogger(__name__)
```

- [ ] **Step 2: Add logging to all 4 blocks**

**Note:** Line numbers below are pre-insertion values. After adding the import (+3 lines), each shifts by +3.

Line 81-82 — telemetry store init:
```python
        except Exception as exc:
            logger.debug("Telemetry store init failed: %s", exc)
```

Line 108-109 — energy monitor creation:
```python
        except Exception as exc:
            logger.debug("Energy monitor creation failed: %s", exc)
```

Line 112-113 — instrumented engine wrapping:
```python
        except Exception as exc:
            logger.debug("Engine instrumentation failed: %s", exc)
```

Line 209-210 — speech backend discovery:
```python
        except Exception as exc:
            logger.debug("Speech backend discovery failed: %s", exc)
```

- [ ] **Step 3: Lint check**

Run: `uv run ruff check src/openjarvis/cli/serve.py`
Expected: All checks passed

- [ ] **Step 4: Commit**

```bash
git add src/openjarvis/cli/serve.py
git commit -m "fix: add debug logging to silent exception handlers in cli/serve.py"
```

---

### Task 5: Add logging to `server/api_routes.py`

**Files:**
- Modify: `src/openjarvis/server/api_routes.py:1-10` (add import)
- Modify: 11 exception blocks (lines 66, 284, 399, 539, 552, 565, 576, 610, 627, 735, 766)

- [ ] **Step 1: Add logger setup**

After `from pydantic import BaseModel` (line 10), before the `# ---- Request/Response models ----` comment (line 12), add:
```python

import logging

logger = logging.getLogger(__name__)

```

**Note:** All line numbers below are pre-insertion values. After adding the import (+4 lines), each shifts by +4.

- [ ] **Step 2: Add warning logging to each block**

Line 66-67 — agent registry listing:
```python
    except Exception as exc:
        logger.warning("Failed to list registered agents: %s", exc)
```

Line 284-285 — skills listing:
```python
    except Exception as exc:
        logger.warning("Failed to list skills: %s", exc)
        return {"skills": []}
```

Line 399-403 — Prometheus metrics:
```python
    except Exception as exc:
        logger.warning("Failed to collect Prometheus metrics: %s", exc)
        from starlette.responses import PlainTextResponse
        return PlainTextResponse(
```

Line 539-540 — GRPO stats:
```python
    except Exception as exc:
        logger.warning("Failed to load GRPO stats: %s", exc)
        result["grpo"] = {"available": False}
```

Line 552-553 — bandit stats:
```python
    except Exception as exc:
        logger.warning("Failed to load bandit stats: %s", exc)
        result["bandit"] = {"available": False}
```

Line 565-566 — ICL stats:
```python
    except Exception as exc:
        logger.warning("Failed to load ICL stats: %s", exc)
        result["icl"] = {"available": False}
```

Line 576-577 — skill discovery stats:
```python
    except Exception as exc:
        logger.warning("Failed to load skill discovery stats: %s", exc)
        result["skill_discovery"] = {"available": False}
```

Line 610-611 — learning config:
```python
    except Exception as exc:
        logger.warning("Failed to load learning config: %s", exc)
        result["enabled"] = False
```

Line 627-628 — GRPO weights:
```python
        except Exception as exc:
            logger.warning("Failed to load GRPO weights: %s", exc)
            result["grpo_weights"] = {}
```

Line 735-736 — optimization runs:
```python
    except Exception as exc:
        logger.warning("Failed to list optimization runs: %s", exc)
        return {"runs": []}
```

Line 766-767 — optimization run detail:
```python
    except Exception as exc:
        logger.warning("Failed to get optimization run %s: %s", run_id, exc)
        return {"run_id": run_id, "status": "not_found"}
```

- [ ] **Step 3: Lint check**

Run: `uv run ruff check src/openjarvis/server/api_routes.py`
Expected: All checks passed

- [ ] **Step 4: Commit**

```bash
git add src/openjarvis/server/api_routes.py
git commit -m "fix: add warning logging to silent exception handlers in api_routes.py"
```

---

### Task 6: Add logging to tools modules

**Files:**
- Modify: `src/openjarvis/tools/agent_tools.py:198-199` (add import + logging)
- Modify: `src/openjarvis/tools/git_tool.py:384-385` (add import + logging)
- Modify: `src/openjarvis/tools/http_request.py:121-122` (add import + logging)
- Modify: `src/openjarvis/tools/templates/loader.py:188-189` (add import + logging)

- [ ] **Step 1: Add logger to each file**

For each of the 4 files, add near the top (after existing imports):
```python
import logging

logger = logging.getLogger(__name__)
```

- [ ] **Step 2: Add debug logging to each block**

`agent_tools.py:198-199` — event bus publish:
```python
        except Exception as exc:
            logger.debug("Event bus publish failed for agent_send: %s", exc)
```

`git_tool.py:384-385` — Rust module fallback:
```python
        except Exception as exc:
            logger.debug("Rust git_log fallback to CLI: %s", exc)
```

`http_request.py:121-122` — Rust HTTP fallback:
```python
        except Exception as exc:
            logger.debug("Rust HTTP request fallback to httpx: %s", exc)
```

`templates/loader.py:188-189` — template parse skip:
```python
        except Exception as exc:
            logger.debug("Skipping unparseable template %s: %s", path, exc)
```

- [ ] **Step 3: Lint check**

Run: `uv run ruff check src/openjarvis/tools/agent_tools.py src/openjarvis/tools/git_tool.py src/openjarvis/tools/http_request.py src/openjarvis/tools/templates/loader.py`
Expected: All checks passed

- [ ] **Step 4: Commit**

```bash
git add src/openjarvis/tools/agent_tools.py src/openjarvis/tools/git_tool.py src/openjarvis/tools/http_request.py src/openjarvis/tools/templates/loader.py
git commit -m "fix: add debug logging to silent exception handlers in tools modules"
```

---

**Out of scope for this PR:** Additional silent exception blocks exist in `agents/__init__.py`, `server/app.py`, `cli/bench_cmd.py`, `cli/optimize_cmd.py`, `operators/manager.py`, `cli/operators_cmd.py`, `cli/dashboard.py`. These are deferred to a follow-up PR.

---

## Final Verification

- [ ] **Full lint check**

Run: `uv run ruff check src/ tests/`
Expected: All checks passed

- [ ] **Full test suite**

Run: `uv run pytest tests/ -m "not live and not cloud and not slow" --tb=short`
Expected: No new failures (pre-existing Rust bridge failures are OK)

- [ ] **Verify no remaining silent blocks in touched files**

Check that every `except Exception` block in the touched files now has a `logger.` call:
```bash
# Find all except-Exception lines in touched files
grep -n "except Exception" src/openjarvis/cli/serve.py src/openjarvis/server/api_routes.py src/openjarvis/tools/agent_tools.py src/openjarvis/tools/git_tool.py src/openjarvis/tools/http_request.py src/openjarvis/tools/templates/loader.py
```
Then for each match, verify the next 1-2 lines contain `logger.` (manual spot check or use `grep -A2`).

- [ ] **Verify identity fix**

Run: `grep "pillar" pyproject.toml`
Expected: No output
