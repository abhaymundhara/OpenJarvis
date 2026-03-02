# Codebase Simplification Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Simplify the OpenJarvis repository structure by reducing root sprawl, moving data files into the package, removing backward-compat shims, and consolidating repetitive code patterns.

**Architecture:** Incremental commits, each independently testable. Data TOML files move into `src/openjarvis/*/data/` as package data. The `evals/` framework becomes `src/openjarvis/evals/` with all internal imports rewritten from `evals.` to `openjarvis.evals.`. The `memory/` shim package is removed. Engine wrappers are consolidated.

**Tech Stack:** Python, TOML, hatch/hatchling build system, pytest, ruff

---

### Task 1: Remove get-pip.py from git tracking

**Files:**
- Delete: `get-pip.py`
- Modify: `.gitignore`

**Step 1: Add get-pip.py to .gitignore**

In `.gitignore`, add after the `# Project` section (after line 44):

```
get-pip.py
```

**Step 2: Remove get-pip.py from git tracking and delete**

Run:
```bash
git rm get-pip.py
```

**Step 3: Commit**

```bash
git add .gitignore
git commit -m "chore: remove get-pip.py from repository (2.2MB regenerable artifact)"
```

---

### Task 2: Move Docker files to deploy/docker/

**Files:**
- Move: `Dockerfile` -> `deploy/docker/Dockerfile`
- Move: `Dockerfile.gpu` -> `deploy/docker/Dockerfile.gpu`
- Move: `Dockerfile.gpu.rocm` -> `deploy/docker/Dockerfile.gpu.rocm`
- Move: `Dockerfile.sandbox` -> `deploy/docker/Dockerfile.sandbox`
- Move: `docker-compose.yml` -> `deploy/docker/docker-compose.yml`
- Move: `docker-compose.gpu.rocm.yml` -> `deploy/docker/docker-compose.gpu.rocm.yml`
- Modify: `deploy/docker/docker-compose.yml` (update dockerfile paths)
- Modify: `deploy/docker/docker-compose.gpu.rocm.yml` (update dockerfile path)

**Step 1: Move all Docker files**

```bash
mv Dockerfile deploy/docker/Dockerfile
mv Dockerfile.gpu deploy/docker/Dockerfile.gpu
mv Dockerfile.gpu.rocm deploy/docker/Dockerfile.gpu.rocm
mv Dockerfile.sandbox deploy/docker/Dockerfile.sandbox
mv docker-compose.yml deploy/docker/docker-compose.yml
mv docker-compose.gpu.rocm.yml deploy/docker/docker-compose.gpu.rocm.yml
```

**Step 2: Update docker-compose.yml build contexts**

In `deploy/docker/docker-compose.yml`, change the `build` section from:
```yaml
    build:
      context: .
      dockerfile: Dockerfile
```
to:
```yaml
    build:
      context: ../..
      dockerfile: deploy/docker/Dockerfile
```

**Step 3: Update docker-compose.gpu.rocm.yml**

In `deploy/docker/docker-compose.gpu.rocm.yml`, change:
```yaml
      dockerfile: Dockerfile.gpu.rocm
```
to:
```yaml
      context: ../..
      dockerfile: deploy/docker/Dockerfile.gpu.rocm
```

**Step 4: Commit**

```bash
git add -A deploy/docker/
git add -A Dockerfile* docker-compose*
git commit -m "refactor: move Docker files to deploy/docker/"
```

---

### Task 3: Move recipes/ TOML data into package

**Files:**
- Move: `recipes/*.toml` -> `src/openjarvis/recipes/data/*.toml`
- Move: `recipes/operators/*.toml` -> `src/openjarvis/recipes/data/operators/*.toml`
- Move: `recipes/operators/*.md` -> `src/openjarvis/recipes/data/operators/*.md`
- Modify: `src/openjarvis/recipes/loader.py` (lines 15-16, update `_PROJECT_RECIPES_DIR`)
- Modify: `pyproject.toml` (add package data include)
- Test: `tests/recipes/` (run existing tests)

**Step 1: Create data directory and move files**

```bash
mkdir -p src/openjarvis/recipes/data/operators
mv recipes/*.toml src/openjarvis/recipes/data/
mv recipes/operators/*.toml src/openjarvis/recipes/data/operators/
mv recipes/operators/*.md src/openjarvis/recipes/data/operators/
rmdir recipes/operators
rmdir recipes
```

**Step 2: Update loader.py**

In `src/openjarvis/recipes/loader.py`, change line 16 from:
```python
_PROJECT_RECIPES_DIR = Path(__file__).resolve().parents[3] / "recipes"
```
to:
```python
_PROJECT_RECIPES_DIR = Path(__file__).resolve().parent / "data"
```

**Step 3: Run tests**

```bash
uv run pytest tests/recipes/ -v
```
Expected: All recipe tests pass.

**Step 4: Commit**

```bash
git add src/openjarvis/recipes/data/ src/openjarvis/recipes/loader.py
git add recipes/
git commit -m "refactor: move recipes/ TOML data into src/openjarvis/recipes/data/"
```

---

### Task 4: Move templates/ agent TOML data into package

**Files:**
- Move: `templates/agents/*.toml` -> `src/openjarvis/templates/data/*.toml`
- Modify: `src/openjarvis/templates/agent_templates.py` (lines 68-71, update `_builtin_templates_dir`)
- Test: `tests/templates/` (run existing tests)

**Step 1: Create data directory and move files**

```bash
mkdir -p src/openjarvis/templates/data
mv templates/agents/*.toml src/openjarvis/templates/data/
rmdir templates/agents
rmdir templates
```

**Step 2: Update agent_templates.py**

In `src/openjarvis/templates/agent_templates.py`, change lines 68-71 from:
```python
def _builtin_templates_dir() -> Path:
    """Return the path to the built-in templates shipped with the package."""
    # templates/agents/ at project root, 3 levels above this file
    return Path(__file__).resolve().parents[3] / "templates" / "agents"
```
to:
```python
def _builtin_templates_dir() -> Path:
    """Return the path to the built-in templates shipped with the package."""
    return Path(__file__).resolve().parent / "data"
```

**Step 3: Run tests**

```bash
uv run pytest tests/templates/ -v
```
Expected: All template tests pass.

**Step 4: Commit**

```bash
git add src/openjarvis/templates/data/ src/openjarvis/templates/agent_templates.py
git add templates/
git commit -m "refactor: move templates/agents/ TOML data into src/openjarvis/templates/data/"
```

---

### Task 5: Move skills/builtin/ TOML data into package

**Files:**
- Move: `skills/builtin/*.toml` -> `src/openjarvis/skills/data/*.toml`
- Modify: `tests/skills/test_bundled_skills.py` (line 12, update `BUILTIN_DIR` path)
- Test: `tests/skills/` (run existing tests)

**Step 1: Create data directory and move files**

```bash
mkdir -p src/openjarvis/skills/data
mv skills/builtin/*.toml src/openjarvis/skills/data/
rmdir skills/builtin
rmdir skills
```

**Step 2: Update test_bundled_skills.py**

In `tests/skills/test_bundled_skills.py`, change line 12 from:
```python
BUILTIN_DIR = Path(__file__).resolve().parents[2] / "skills" / "builtin"
```
to:
```python
BUILTIN_DIR = Path(__file__).resolve().parents[2] / "src" / "openjarvis" / "skills" / "data"
```

**Step 3: Run tests**

```bash
uv run pytest tests/skills/ -v
```
Expected: All skill tests pass (20 bundled skills found).

**Step 4: Commit**

```bash
git add src/openjarvis/skills/data/ tests/skills/test_bundled_skills.py
git add skills/
git commit -m "refactor: move skills/builtin/ TOML data into src/openjarvis/skills/data/"
```

---

### Task 6: Move operators/ TOML data into package

**Files:**
- Move: `operators/*.toml` -> `src/openjarvis/operators/data/*.toml`
- Modify: `src/openjarvis/cli/operators_cmd.py` (lines 35, 247, 274 — replace `Path("operators")` with package data path)
- Test: `tests/operators/` (run existing tests)

**Step 1: Create data directory and move files**

```bash
mkdir -p src/openjarvis/operators/data
mv operators/*.toml src/openjarvis/operators/data/
rmdir operators
```

**Step 2: Update operators_cmd.py**

The CLI currently checks `Path("operators")` (cwd-relative). We need to replace this with the package data directory. In `src/openjarvis/cli/operators_cmd.py`:

Add a helper at the top of the file (after imports):
```python
def _builtin_operators_dir() -> Path:
    """Return the path to built-in operator manifests shipped with the package."""
    return Path(__file__).resolve().parents[1] / "operators" / "data"
```

Then update the three places that reference `Path("operators")`:

Line 35: Change `local_ops = Path("operators")` to `local_ops = _builtin_operators_dir()`

Line 247: Change `dirs = [DEFAULT_CONFIG_DIR / "operators", Path("operators")]` to `dirs = [DEFAULT_CONFIG_DIR / "operators", _builtin_operators_dir()]`

Line 274: Change `for d in [DEFAULT_CONFIG_DIR / "operators", Path("operators")]:` to `for d in [DEFAULT_CONFIG_DIR / "operators", _builtin_operators_dir()]:`

**Step 3: Run tests**

```bash
uv run pytest tests/operators/ -v
```
Expected: All operator tests pass.

**Step 4: Commit**

```bash
git add src/openjarvis/operators/data/ src/openjarvis/cli/operators_cmd.py
git add operators/
git commit -m "refactor: move operators/ TOML data into src/openjarvis/operators/data/"
```

---

### Task 7: Move evals/ into src/openjarvis/evals/

This is the most complex migration. The `evals/` directory is a standalone Python package with internal imports like `from evals.core.config import ...`. All these must be rewritten to `from openjarvis.evals.core.config import ...`.

**Files:**
- Move: `evals/` -> `src/openjarvis/evals/` (entire directory tree)
- Modify: All `*.py` files in `src/openjarvis/evals/` (rewrite `from evals.` to `from openjarvis.evals.`)
- Modify: `src/openjarvis/cli/eval_cmd.py` (rewrite `from evals.` imports)
- Modify: `tests/evals/*.py` (rewrite `from evals.` imports)
- Modify: `pyproject.toml` (update ruff per-file-ignores paths)

**Step 1: Move the directory**

```bash
# Move evals/ into src/openjarvis/evals/
# First remove any existing __pycache__
find evals/ -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null
mv evals/ src/openjarvis/evals/
```

**Step 2: Rewrite all internal imports in evals/**

Run a find-and-replace across all Python files in `src/openjarvis/evals/`:
```bash
find src/openjarvis/evals/ -name "*.py" -exec sed -i 's/from evals\./from openjarvis.evals./g; s/import evals\./import openjarvis.evals./g' {} +
```

**Step 3: Rewrite imports in eval_cmd.py**

In `src/openjarvis/cli/eval_cmd.py`, replace all occurrences of:
- `from evals.core.config import` -> `from openjarvis.evals.core.config import`
- `from evals.cli import` -> `from openjarvis.evals.cli import`
- `from evals.core.types import` -> `from openjarvis.evals.core.types import`

**Step 4: Rewrite imports in tests/evals/**

Run find-and-replace across test files:
```bash
find tests/evals/ -name "*.py" -exec sed -i 's/from evals\./from openjarvis.evals./g; s/import evals\./import openjarvis.evals./g' {} +
```

**Step 5: Update pyproject.toml**

Change lines 134-136 from:
```toml
[tool.ruff.lint.per-file-ignores]
"evals/datasets/*.py" = ["E501"]
"evals/scorers/*.py" = ["E501"]
```
to:
```toml
[tool.ruff.lint.per-file-ignores]
"src/openjarvis/evals/datasets/*.py" = ["E501"]
"src/openjarvis/evals/scorers/*.py" = ["E501"]
```

**Step 6: Update __main__.py**

In `src/openjarvis/evals/__main__.py`, the import was rewritten in step 2 but verify it now reads:
```python
"""Allow running as ``python -m openjarvis.evals``."""

from openjarvis.evals.cli import main

if __name__ == "__main__":
    main()
```

**Step 7: Run tests**

```bash
uv run pytest tests/evals/ -v
```
Expected: All eval tests pass.

**Step 8: Lint check**

```bash
uv run ruff check src/openjarvis/evals/ tests/evals/ src/openjarvis/cli/eval_cmd.py
```
Expected: No import errors.

**Step 9: Commit**

```bash
git add src/openjarvis/evals/ tests/evals/ src/openjarvis/cli/eval_cmd.py pyproject.toml
git commit -m "refactor: move evals/ into src/openjarvis/evals/ as proper subpackage"
```

---

### Task 8: Remove memory/ backward-compat shim package

**Files:**
- Delete: `src/openjarvis/memory/` (entire directory, 11 files)
- Modify: `src/openjarvis/cli/memory_cmd.py` (lines 15-16)
- Modify: `src/openjarvis/cli/ask.py` (lines 130, 304)
- Modify: `src/openjarvis/sdk.py` (lines 36, 59-60, 391, 431)
- Modify: `tests/tools/test_storage_stubs.py` (lines 68-69, 80)
- Modify: `tests/tools/test_retrieval.py` (line 7)
- Modify: `tests/cli/test_memory_cmd.py` (line 12)
- Modify: `tests/cli/test_ask_context.py` (lines 34, 55)
- Modify: `tests/test_integration_extended.py` (lines 402, 419)
- Delete: `tests/memory/` (entire directory — these test the shim, not the real backends)

**Step 1: Update src/ imports**

In `src/openjarvis/cli/memory_cmd.py`, change:
```python
from openjarvis.memory.chunking import ChunkConfig
from openjarvis.memory.ingest import ingest_path
```
to:
```python
from openjarvis.tools.storage.chunking import ChunkConfig
from openjarvis.tools.storage.ingest import ingest_path
```

In `src/openjarvis/cli/ask.py`, change all occurrences of:
- `from openjarvis.memory.context import` -> `from openjarvis.tools.storage.context import`
- `from openjarvis.memory.sqlite import` -> `from openjarvis.tools.storage.sqlite import`

In `src/openjarvis/sdk.py`, change:
- `from openjarvis.memory.sqlite import SqliteMemory` -> `from openjarvis.tools.storage.sqlite import SQLiteMemory as SqliteMemory`
- `from openjarvis.memory.chunking import ChunkConfig` -> `from openjarvis.tools.storage.chunking import ChunkConfig`
- `from openjarvis.memory.ingest import ingest_path` -> `from openjarvis.tools.storage.ingest import ingest_path`
- `from openjarvis.memory.context import ContextConfig, inject_context` -> `from openjarvis.tools.storage.context import ContextConfig, inject_context`

**Step 2: Update test imports**

In `tests/tools/test_storage_stubs.py`, change:
- `from openjarvis.memory._stubs import MemoryBackend as MB` -> `from openjarvis.tools.storage._stubs import MemoryBackend as MB`
- `from openjarvis.memory._stubs import RetrievalResult as RR` -> `from openjarvis.tools.storage._stubs import RetrievalResult as RR`
- `from openjarvis.memory.sqlite import SQLiteMemory as S2` -> `from openjarvis.tools.storage.sqlite import SQLiteMemory as S2`

In `tests/tools/test_retrieval.py`, change:
- `from openjarvis.memory._stubs import MemoryBackend, RetrievalResult` -> `from openjarvis.tools.storage._stubs import MemoryBackend, RetrievalResult`

In `tests/cli/test_memory_cmd.py`, change:
- `from openjarvis.memory.sqlite import SQLiteMemory` -> `from openjarvis.tools.storage.sqlite import SQLiteMemory`

In `tests/cli/test_ask_context.py`, change:
- `from openjarvis.memory.sqlite import SQLiteMemory` -> `from openjarvis.tools.storage.sqlite import SQLiteMemory`

In `tests/test_integration_extended.py`, change:
- `from openjarvis.memory.sqlite import SQLiteMemory` -> `from openjarvis.tools.storage.sqlite import SQLiteMemory`
- `from openjarvis.memory.bm25 import BM25Memory` -> `from openjarvis.tools.storage.bm25 import BM25Memory`

**Step 3: Update tests/memory/ imports and move to tests/tools/**

The `tests/memory/` files test the same backends as `tests/tools/test_storage_*.py`. Since they import from the shim, we need to rewrite their imports and merge them. However, since `tests/tools/` already has storage tests, the simplest approach is:

Rewrite all `tests/memory/*.py` imports from `openjarvis.memory.*` to `openjarvis.tools.storage.*`:
```bash
find tests/memory/ -name "*.py" -exec sed -i 's/from openjarvis\.memory\./from openjarvis.tools.storage./g' {} +
```

Then move the directory to be a subdirectory of tests/tools:
```bash
mv tests/memory tests/tools/memory_tests
```

Actually, the cleanest approach: just rewrite imports in-place and keep `tests/memory/` as-is. The tests are valuable and test different aspects than `tests/tools/`.

**Step 4: Delete the shim package**

```bash
rm -rf src/openjarvis/memory/
```

**Step 5: Run full test suite**

```bash
uv run pytest tests/memory/ tests/tools/ tests/cli/ tests/sdk/ -v
```
Expected: All tests pass.

**Step 6: Commit**

```bash
git add -A src/openjarvis/memory/ src/openjarvis/cli/memory_cmd.py src/openjarvis/cli/ask.py src/openjarvis/sdk.py
git add tests/memory/ tests/tools/ tests/cli/ tests/test_integration_extended.py
git commit -m "refactor: remove memory/ backward-compat shims, use tools.storage directly"
```

---

### Task 9: Consolidate engine wrappers into single file

**Files:**
- Delete: `src/openjarvis/engine/vllm.py`
- Delete: `src/openjarvis/engine/sglang.py`
- Delete: `src/openjarvis/engine/llamacpp.py`
- Delete: `src/openjarvis/engine/mlx.py`
- Delete: `src/openjarvis/engine/lmstudio.py`
- Create: `src/openjarvis/engine/openai_compat_engines.py`
- Modify: `src/openjarvis/engine/__init__.py` (update imports)
- Test: Run `tests/engine/` tests

**Step 1: Write the consolidated engine file**

Create `src/openjarvis/engine/openai_compat_engines.py`:

```python
"""Data-driven registration of OpenAI-compatible inference engines."""

from openjarvis.core.registry import EngineRegistry
from openjarvis.engine._openai_compat import _OpenAICompatibleEngine

_ENGINES = {
    "vllm": ("VLLMEngine", "http://localhost:8000"),
    "sglang": ("SGLangEngine", "http://localhost:30000"),
    "llamacpp": ("LlamaCppEngine", "http://localhost:8080"),
    "mlx": ("MLXEngine", "http://localhost:8080"),
    "lmstudio": ("LMStudioEngine", "http://localhost:1234"),
}

for _key, (_cls_name, _default_host) in _ENGINES.items():
    _cls = type(
        _cls_name,
        (_OpenAICompatibleEngine,),
        {"engine_id": _key, "_default_host": _default_host},
    )
    EngineRegistry.register(_key)(_cls)
    globals()[_cls_name] = _cls

__all__ = [name for name, _ in _ENGINES.values()]
```

**Step 2: Update engine/__init__.py**

Replace lines 5-11 (the 5 individual imports) with a single import:

Change from:
```python
import openjarvis.engine.llamacpp  # noqa: F401
import openjarvis.engine.mlx  # noqa: F401

# Import engine modules to trigger @EngineRegistry.register() decorators
import openjarvis.engine.ollama  # noqa: F401
import openjarvis.engine.sglang  # noqa: F401
import openjarvis.engine.vllm  # noqa: F401
```

To:
```python
# Import engine modules to trigger @EngineRegistry.register() decorators
import openjarvis.engine.ollama  # noqa: F401
import openjarvis.engine.openai_compat_engines  # noqa: F401
```

**Step 3: Delete the 5 individual wrapper files**

```bash
rm src/openjarvis/engine/vllm.py
rm src/openjarvis/engine/sglang.py
rm src/openjarvis/engine/llamacpp.py
rm src/openjarvis/engine/mlx.py
rm src/openjarvis/engine/lmstudio.py
```

**Step 4: Run tests**

```bash
uv run pytest tests/engine/ -v
```
Expected: All engine tests pass.

**Step 5: Commit**

```bash
git add src/openjarvis/engine/
git commit -m "refactor: consolidate 5 OpenAI-compat engine wrappers into single data-driven file"
```

---

### Task 10: Refactor repetitive __init__.py imports

**Files:**
- Modify: `src/openjarvis/channels/__init__.py`
- Modify: `src/openjarvis/engine/__init__.py` (optional engines section)
- Test: Run `tests/channels/` and `tests/engine/` tests

**Step 1: Refactor channels/__init__.py**

Replace the entire file content with:

```python
"""Channel abstraction for multi-platform messaging."""

import importlib

from openjarvis.channels._stubs import (
    BaseChannel,
    ChannelHandler,
    ChannelMessage,
    ChannelStatus,
)

# Trigger registration of built-in channels.
# Each module uses @ChannelRegistry.register() — importing is sufficient.
_CHANNEL_MODULES = [
    "telegram",
    "discord_channel",
    "slack",
    "webhook",
    "email_channel",
    "whatsapp",
    "signal_channel",
    "google_chat",
    "irc_channel",
    "webchat",
    "teams",
    "matrix_channel",
    "mattermost",
    "feishu",
    "bluebubbles",
    "whatsapp_baileys",
    "line_channel",
    "viber_channel",
    "messenger_channel",
    "reddit_channel",
    "mastodon_channel",
    "xmpp_channel",
    "rocketchat_channel",
    "zulip_channel",
    "twitch_channel",
    "nostr_channel",
]

for _mod in _CHANNEL_MODULES:
    try:
        importlib.import_module(f".{_mod}", __name__)
    except ImportError:
        pass

__all__ = [
    "BaseChannel",
    "ChannelHandler",
    "ChannelMessage",
    "ChannelStatus",
]
```

**Step 2: Refactor optional engine imports in engine/__init__.py**

Replace lines 19-29 with:

```python
# Optional engines — only register if their SDK deps are present
for _optional in ("cloud", "litellm"):
    try:
        importlib.import_module(f".{_optional}", __name__)
    except ImportError:
        pass
```

And add `import importlib` at the top.

**Step 3: Run tests**

```bash
uv run pytest tests/channels/ tests/engine/ -v
```
Expected: All tests pass.

**Step 4: Commit**

```bash
git add src/openjarvis/channels/__init__.py src/openjarvis/engine/__init__.py
git commit -m "refactor: replace repetitive try/except imports with loop-based auto-registration"
```

---

### Task 11: Update pyproject.toml package data includes

**Files:**
- Modify: `pyproject.toml`

**Step 1: Add package data includes**

The TOML data files that moved into `src/openjarvis/` need to be included in wheel builds. In `pyproject.toml`, after the existing `[tool.hatch.build.targets.wheel.force-include]` section, add the data directories.

Actually, since all the data directories are under `src/openjarvis/` and `packages = ["src/openjarvis"]` is already set, hatch will include them automatically as long as they are inside the package tree. But TOML files are non-Python files, so we need to tell hatch to include them.

Add to pyproject.toml after line 114:

```toml
[tool.hatch.build.targets.wheel.shared-data]

[tool.hatch.build]
include = [
    "src/openjarvis/**/*.py",
    "src/openjarvis/**/*.toml",
    "src/openjarvis/**/*.md",
    "src/openjarvis/agents/claude_code_runner/**",
    "src/openjarvis/channels/whatsapp_baileys_bridge/**",
]
```

Wait — hatchling by default includes all files under `packages`. Non-Python files inside the package are included by default. No change needed here. Verify with:

```bash
uv run python -c "from pathlib import Path; d = Path('src/openjarvis/recipes/data'); print(list(d.glob('*.toml')))"
```

**Step 2: Commit if any changes were needed**

If no pyproject.toml changes needed, skip this task.

---

### Task 12: Update CLAUDE.md to reflect new structure

**Files:**
- Modify: `CLAUDE.md`

**Step 1: Update all references to moved paths**

In CLAUDE.md, make the following updates:

1. Update the `python -m evals` command references to `python -m openjarvis.evals`
2. Remove references to top-level `recipes/`, `templates/`, `skills/`, `operators/` directories
3. Update Docker command references to use `deploy/docker/` paths
4. Update the architecture section to reflect that data files are now package data
5. Remove mentions of `memory/` backward-compat shims
6. Remove the 5 individual engine wrapper files from architecture description

**Step 2: Run linting to verify no broken references**

```bash
uv run ruff check src/ tests/
```

**Step 3: Run full test suite**

```bash
uv run pytest tests/ -v --tb=short 2>&1 | tail -20
```
Expected: ~3240 tests pass.

**Step 4: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md to reflect simplified repository structure"
```

---

### Task 13: Final verification

**Step 1: Run full test suite**

```bash
uv run pytest tests/ -v --tb=short 2>&1 | tail -30
```
Expected: ~3240 tests pass, ~44 skipped.

**Step 2: Run linter**

```bash
uv run ruff check src/ tests/
```
Expected: Clean.

**Step 3: Verify root structure**

```bash
ls -la | grep -v __pycache__ | grep -v '^\.'
```
Expected: 21 or fewer items at root (down from 32).

**Step 4: Verify package data is accessible**

```bash
uv run python -c "
from openjarvis.recipes.loader import discover_recipes
from openjarvis.templates.agent_templates import discover_templates
print(f'Recipes: {len(discover_recipes())}')
print(f'Templates: {len(discover_templates())}')
print('All data accessible.')
"
```
Expected: Recipes: 3, Templates: 15, All data accessible.
