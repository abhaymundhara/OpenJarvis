# OpenJarvis Development Notes

Living document tracking implementation progress, testing state, lessons learned, dead ends, and practices for ongoing development. Updated across sessions.

---

## Current State (2026-03-02)

- **Version:** 1.0.0 (Phase 23 complete)
- **All 23 phases complete** — five composable pillars: Intelligence, Engine, Agents, Tools, Learning
- **Tests:** 3270 passed, 44 skipped, 0 failures
- **Lint:** 117 pre-existing warnings (0 in recently changed files)
- **Source files:** ~251 Python files in `src/openjarvis/`
- **Test files:** ~263 Python files in `tests/`
- **Python:** 3.13 (compatible with 3.10+)
- **Package manager:** `uv` with `hatchling` build backend
- **Root items:** 20 (down from 32 after Session 4 cleanup)

### 8 Skipped Tests (Optional Dependencies)

| Test | Missing Dep | Install Extra |
|------|-------------|---------------|
| `tests/memory/test_bm25.py` | `rank_bm25` | `openjarvis[memory-bm25]` |
| `tests/memory/test_colbert.py` | `colbert` | `openjarvis[memory-colbert]` |
| `tests/memory/test_embeddings.py` | `sentence_transformers` | `openjarvis[memory-faiss]` |
| `tests/memory/test_faiss.py` | `faiss` | `openjarvis[memory-faiss]` |
| `tests/server/test_models_pydantic.py` | `pydantic` | `openjarvis[server]` |
| `tests/server/test_routes.py` | `fastapi` | `openjarvis[server]` |
| `tests/test_integration.py:165` | `fastapi` | `openjarvis[server]` |
| `tests/test_integration.py:190` | `fastapi` | `openjarvis[server]` |

---

## Phase Completion Log

| Phase | Version | Deliverables | Test Count (cumulative) |
|-------|---------|-------------|------------------------|
| Phase 0 | v0.1 | Scaffolding, registries, core types, config, CLI skeleton, event bus | ~60 |
| Phase 1 | v0.2 | Intelligence + Inference — `jarvis ask` end-to-end, heuristic router, engine discovery, basic telemetry | ~160 |
| Phase 2 | v0.3 | Memory — SQLite/FAISS/ColBERT/BM25/Hybrid backends, document ingest pipeline, context injection, `jarvis memory` CLI | ~270 |
| Phase 3 | v0.4 | Agents (Simple/Orchestrator/Custom/OpenClaw stub), tool system (Calculator/Think/Retrieval/LLM/FileRead), OpenAI-compatible API server, `jarvis serve` | ~360 |
| Phase 4 | v0.5 | Learning — HeuristicRouter, HeuristicRewardFunction, GRPORouterPolicy stub, TelemetryAggregator, `jarvis telemetry` CLI, `--router` CLI option | ~432 |
| Phase 5 | v1.0 | SDK (`Jarvis` class), OpenClaw infrastructure (protocol/transport/plugin), benchmarks (`jarvis bench`), Docker, docs | ~520 |
| Phase 6 | v1.1 | Trace system (TraceStore, TraceCollector, TraceAnalyzer), trace-driven learning (TraceDrivenPolicy) | ~576 |
| Phase 7 | v1.2 | 5-pillar restructuring, composition layer (SystemBuilder/JarvisSystem), MCP, structured learning | — |
| Phase 8 | v1.3 | Intelligence = "The Model", routing → Learning pillar, engine selection | — |
| Phase 9 | v1.4 | Pillar-aligned config, nested TOML configs, config migration | — |
| Phase 10 | v1.5 | Agent restructuring (BaseAgent/ToolUsingAgent), `accepts_tools`, OpenHands SDK | — |
| Phase 11 | v1.6 | NanoClaw subsumption: ClaudeCodeAgent, WhatsApp Baileys, Docker sandbox, TaskScheduler | — |
| Phase 12 | v1.7 | EnergyMonitor ABC (NVIDIA/AMD/Apple/RAPL), EnergyBatch, SteadyStateDetector | — |
| Phase 13 | v1.8 | `jarvis doctor`/`init`, MLX engine, AMD multi-GPU, PWA, ROCm Docker | — |
| Phase 14 | v1.9 | Agent hardening: LoopGuard, RBAC CapabilityPolicy, taint tracking, Merkle audit, Ed25519 | — |
| Phase 15 | v2.0 | WorkflowEngine (DAG), SkillSystem, KnowledgeGraphMemory, SessionStore | — |
| Phase 16 | v2.1 | A2A protocol, MCP templates, WasmRunner, TUI dashboard | — |
| Phase 17 | v2.2 | Production tool parity: 40+ tools, SSRF/injection/rate-limit, security middleware | — |
| Phase 18 | v2.3 | CLI expansion (20 commands), API expansion (40+ endpoints), WebSocket streaming | — |
| Phase 19 | v2.4 | Learning productionization: GRPO, BanditRouter, SkillDiscovery, ICL updates | — |
| Phase 20 | v2.5 | Tauri 2.0 desktop app (5 dashboard panels), CI for Linux/macOS/Windows | — |
| Phase 21 | v2.6 | 10 new channels: LINE, Viber, Messenger, Reddit, Mastodon, XMPP, Rocket.Chat, Zulip, Twitch, Nostr | ~2940 |
| Phase 22 | v2.7 | Operators: persistent, scheduled autonomous agents with recipe + schedule + channel output | ~2997 |
| Phase 23 | v2.8 | Differentiated functionalities: trace-driven learning pipeline, 15 real IPW benchmarks, composable recipes, 15 agent templates, 20 bundled skills, 3 operator recipes | ~3270 |

---

## Architecture Quick Reference

### Directory Layout

```
src/openjarvis/
├── __init__.py          # __version__ = "1.0.0", exports Jarvis, MemoryHandle
├── sdk.py               # Python SDK: Jarvis class + MemoryHandle
├── system.py            # SystemBuilder + JarvisSystem composition layer
├── core/
│   ├── registry.py      # RegistryBase[T] + 10 typed registries
│   ├── types.py         # Message, Conversation, ModelSpec, ToolResult, TelemetryRecord, Trace
│   ├── config.py        # JarvisConfig dataclass hierarchy, TOML loader
│   └── events.py        # EventBus pub/sub (synchronous)
├── intelligence/        # ModelRegistry, model catalog, generation defaults
├── engine/              # Ollama + openai_compat_engines.py (data-driven: vLLM/SGLang/llama.cpp/MLX/LM Studio) + Cloud
├── agents/              # BaseAgent/ToolUsingAgent hierarchy, 10+ agent types
├── tools/               # 40+ tools via MCP, storage backends (SQLite/FAISS/ColBERT/BM25/Hybrid/KG)
├── learning/            # RouterPolicyRegistry, GRPO, Bandit, SFT, ICL, SkillDiscovery
├── traces/              # TraceStore, TraceCollector, TraceAnalyzer
├── evals/               # 15 IPW benchmarks, EvalRunner, scorer types, CLI
├── recipes/data/        # Composable TOML recipe configs
├── templates/data/      # 15 agent template TOML manifests
├── skills/data/         # 20 bundled skill TOML manifests
├── operators/data/      # 3 operator recipe TOML manifests
├── channels/            # 25+ messaging channel backends
├── bench/               # Latency/Throughput/Energy benchmarks
├── telemetry/           # TelemetryStore, EnergyMonitor, InstrumentedEngine
├── security/            # Scanners, RBAC, audit, taint tracking, sandboxing
├── server/              # FastAPI OpenAI-compatible API server (40+ endpoints)
├── scheduler/           # Cron/interval task scheduling
├── workflow/            # DAG-based workflow engine
├── sessions/            # Cross-channel persistent sessions
├── sandbox/             # Docker/Wasm sandboxed execution
├── a2a/                 # Google Agent-to-Agent protocol
└── cli/                 # Click CLI: 20+ subcommands
```

### 7 Registries

All use `RegistryBase[T]` with `@XRegistry.register("name")` or `register_value()`:

1. `ModelRegistry` — `ModelSpec` objects
2. `EngineRegistry` — `InferenceEngine` implementations
3. `MemoryRegistry` — `MemoryBackend` implementations
4. `AgentRegistry` — `BaseAgent` implementations
5. `ToolRegistry` — `BaseTool` implementations
6. `RouterPolicyRegistry` — `RouterPolicy` implementations
7. `BenchmarkRegistry` — `BaseBenchmark` implementations

---

## Patterns and Practices

### The `ensure_registered()` Pattern

**Problem:** The `_clean_registries` autouse fixture in `tests/conftest.py` calls `.clear()` on every registry before each test. Module-level `@XRegistry.register("name")` decorators only fire once at import time (Python caches modules in `sys.modules`). After registry clearing, the decorations never re-fire, leaving registries empty for subsequent tests.

**Solution:** Use lazy registration via `ensure_registered()`:

```python
# src/openjarvis/bench/latency.py
_registered = False

def ensure_registered() -> None:
    global _registered
    if _registered:
        return
    from openjarvis.core.registry import BenchmarkRegistry
    if not BenchmarkRegistry.contains("latency"):
        BenchmarkRegistry.register_value("latency", LatencyBenchmark)
    _registered = True
```

Then in `__init__.py`:
```python
def ensure_registered() -> None:
    from openjarvis.bench.latency import ensure_registered as _reg_latency
    _reg_latency()
```

And in test files, use an autouse fixture:
```python
@pytest.fixture(autouse=True)
def _register_latency():
    from openjarvis.bench import ensure_registered
    ensure_registered()
```

**Where this pattern is used:** `bench/latency.py`, `bench/throughput.py`, `learning/heuristic_policy.py`, `learning/grpo_policy.py`, `learning/heuristic_reward.py`

**Where this pattern is NOT needed:** Agents, engines, memory backends, and tools use `@register` decorators that work fine because their test files explicitly import and re-register as needed, or the test module import triggers registration.

### Test Infrastructure

- **`tests/conftest.py`** — `_clean_registries` autouse fixture clears all 7 registries + clears `EventBus` default listeners before each test. Critical for test isolation.
- **Mock engine pattern** — Almost every test that touches the engine layer uses a `MagicMock()` with `.engine_id`, `.health()`, `.list_models()`, `.generate()` stubbed:
  ```python
  def _make_engine(content="Hello"):
      engine = MagicMock()
      engine.engine_id = "mock"
      engine.health.return_value = True
      engine.list_models.return_value = ["test-model"]
      engine.generate.return_value = {
          "content": content,
          "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
          "model": "test-model",
          "finish_reason": "stop",
      }
      return engine
  ```
- **CLI tests** use Click's `CliRunner` with `patch("openjarvis.cli.X.get_engine", ...)` to mock the engine layer.
- **Memory tests** use `tmp_path` fixture for SQLite DB paths and test files.
- **Optional dep tests** use `pytest.importorskip("module_name")` at module level.

### Config Defaults

`JarvisConfig()` with no arguments produces sane defaults:
- Engine: auto-discover (Ollama, vLLM, llama.cpp, cloud in priority order)
- Memory: `sqlite` backend, `~/.openjarvis/memory.db`
- Agent: `simple` (no Node.js dependency)
- Intelligence: `qwen3:8b` default, `qwen3:0.6b` fallback
- Telemetry: enabled, `~/.openjarvis/telemetry.db`
- Learning: `heuristic` default policy

### File Naming Conventions

- ABCs and shared dataclasses: `_stubs.py` (e.g., `agents/_stubs.py`, `bench/_stubs.py`, `tools/_stubs.py`)
- Internal helpers: `_discovery.py`, `_base.py` (underscore prefix)
- CLI commands: `*_cmd.py` (e.g., `bench_cmd.py`, `telemetry_cmd.py`, `memory_cmd.py`)
- Test files mirror source: `tests/agents/test_openclaw.py` tests `src/openjarvis/agents/openclaw.py`

### Import Structure

- Package `__init__.py` files import submodules to trigger registration
- Try/except around optional dependency imports:
  ```python
  try:
      from openjarvis.engine.ollama import OllamaEngine  # noqa: F401
  except ImportError:
      pass
  ```
- Top-level `openjarvis/__init__.py` exports: `Jarvis`, `MemoryHandle`, `__version__`

---

## Dead Ends and Gotchas

### 1. `@register` Decorator vs. `ensure_registered()`

**Dead end:** Initially used `@BenchmarkRegistry.register("latency")` class decorator in `bench/latency.py`. This caused ~10 test failures because:
- Registry cleared between tests by `conftest.py`
- Module already in `sys.modules`, so `import openjarvis.bench` is a no-op on second import
- Registry stays empty after clearing

**Fix:** Switched to `ensure_registered()` pattern (see above). This is the pattern already used by `learning/` modules.

**Rule of thumb:** If a module is imported at package init time AND its registry gets cleared in tests, use `ensure_registered()`. If registration only happens in test fixtures or explicit calls, `@register` is fine.

### 2. Chunk Attribute Names

`memory/chunking.py` `Chunk` dataclass uses `content` (not `text`). `ChunkConfig` uses `chunk_overlap` (not `overlap`). Easy to get wrong because these aren't obvious from the field names alone. Always read `_stubs.py` or the actual dataclass before using.

### 3. Test Content Size for Chunking

`ChunkConfig.min_chunk_size=50` tokens by default. A test string like `"This is test content."` produces 0 chunks. Use at least ~100 words:
```python
words = " ".join(f"word{i}" for i in range(100))
```

### 4. Version String Locations

Version is defined in **three places** that must stay in sync:
1. `src/openjarvis/__init__.py` — `__version__ = "1.0.0"`
2. `pyproject.toml` — `version = "1.0.0"`
3. `src/openjarvis/server/app.py` — FastAPI `version="1.0.0"` constructor arg

Tests that check version: `tests/cli/test_cli.py::test_version_flag`

### 5. Server Import Guards

The `server/` module requires `fastapi`, `uvicorn`, `pydantic`. These are behind the `[server]` optional extra. All test files that touch server code use `pytest.importorskip("fastapi")`. The server `__init__.py` wraps imports in try/except.

### 6. `patch()` Targets for Engine Mocking

When mocking `get_engine` in CLI tests, the patch target must be the *importing module*, not the source module:
```python
# CORRECT — patches where it's imported
patch("openjarvis.cli.bench_cmd.get_engine", return_value=("mock", engine))

# WRONG — patches the source, doesn't affect the already-imported reference
patch("openjarvis.engine._discovery.get_engine", return_value=("mock", engine))
```

Same for SDK tests: `patch("openjarvis.sdk.get_engine", ...)`.

### 7. EventBus Clearing

`EventBus()` creates a new instance each time, but `EventBus._default_listeners` is a class variable. The `conftest.py` fixture resets it. If tests subscribe to events, subscriptions won't persist across tests.

### 8. Module Shadowing in CLI Package

In `cli/__init__.py`, `from openjarvis.cli.ask import ask` imports the Click command. This shadows the module name. When you try `mock.patch("openjarvis.cli.ask.get_engine")`, Python resolves `openjarvis.cli.ask` as the Click command (via attribute lookup on the package), not the module.

**Fix:** Use `importlib.import_module("openjarvis.cli.ask")` to get the actual module object, then `mock.patch.object(module, "get_engine")`.

---

## Post-v1.0: Unimplemented Ideas from VISION.md

These are mentioned in `VISION.md` but not in the roadmap phases. They represent future work:

### Learning / Router
- [ ] Learned router via GRPO (Group Relative Policy Optimization) — `GRPORouterPolicy` is a stub
- [ ] Preference learning from user feedback
- [ ] Continual fine-tuning on accumulated trajectories
- [ ] Multi-objective optimization: quality vs. latency vs. energy vs. cost

### Memory
- [ ] ConversationMemory — sliding window with automatic summarization of older turns
- [ ] Personal Notes — user-created persistent notes and preferences
- [ ] Episodic Memory — records of past interactions, tool uses, and outcomes
- [ ] Vector DB adapters (Qdrant, ChromaDB) for users with existing infrastructure

### Tools
- [ ] WebSearch tool (Tavily, SearXNG, DuckDuckGo)
- [ ] CodeInterpreter tool (sandboxed Python execution)
- [ ] FileWrite tool (safe file writing with path validation)
- [ ] MCP (Model Context Protocol) compatibility

### Engines
- [ ] SGLang engine backend (structured generation, constrained decoding)
- [ ] MLX engine backend (Apple Silicon native, Metal acceleration)
- [ ] Complete vLLM integration (tensor parallelism config, multi-GPU)

### OpenClaw
- [ ] Full OpenClaw gateway integration (WebSocket, `:18789`)
- [ ] OpenClaw skill composition
- [ ] Context compaction in OpenClaw agent
- [ ] `openjarvis-openclaw` as separate plugin package (currently inline)

### Infrastructure
- [ ] Documentation site (MkDocs or similar)
- [ ] Getting started guide
- [ ] Plugin development guide
- [ ] API reference docs
- [ ] CI/CD pipeline
- [ ] PyPI publishing

---

## Testing Recipes

### Run all tests
```bash
uv sync --extra dev
uv run pytest tests/ -v --tb=short
```

### Run a specific module's tests
```bash
uv run pytest tests/bench/ -v
uv run pytest tests/sdk/ -v
uv run pytest tests/agents/test_openclaw.py -v
```

### Run with optional deps (server)
```bash
uv sync --extra dev --extra server
uv run pytest tests/server/ -v  # No longer skipped
```

### Lint
```bash
uv run ruff check src/ tests/
uv run ruff check src/ tests/ --fix  # Auto-fix
```

### Quick smoke test
```bash
uv run jarvis --version          # 1.0.0
uv run jarvis --help             # All subcommands
python -c "from openjarvis import Jarvis; print(Jarvis)"
```

---

## Adding New Components

### New Benchmark

1. Create `src/openjarvis/bench/my_benchmark.py`:
   ```python
   from openjarvis.bench._stubs import BaseBenchmark, BenchmarkResult

   class MyBenchmark(BaseBenchmark):
       @property
       def name(self) -> str: return "my-bench"
       @property
       def description(self) -> str: return "Description"
       def run(self, engine, model, *, num_samples=10) -> BenchmarkResult: ...

   _registered = False
   def ensure_registered():
       global _registered
       if _registered: return
       from openjarvis.core.registry import BenchmarkRegistry
       if not BenchmarkRegistry.contains("my-bench"):
           BenchmarkRegistry.register_value("my-bench", MyBenchmark)
       _registered = True
   ```
2. Import in `bench/__init__.py` `ensure_registered()`
3. Add test file `tests/bench/test_my_benchmark.py` with autouse fixture calling `ensure_registered()`

### New Tool

1. Create `src/openjarvis/tools/my_tool.py`:
   ```python
   from openjarvis.core.registry import ToolRegistry
   from openjarvis.tools._stubs import BaseTool, ToolSpec

   @ToolRegistry.register("my-tool")
   class MyTool(BaseTool):
       @property
       def spec(self) -> ToolSpec: ...
       def execute(self, input: str, **params) -> str: ...
   ```
2. Import in `tools/__init__.py`
3. Add test file `tests/tools/test_my_tool.py`

### New Memory Backend

1. Create `src/openjarvis/tools/storage/my_backend.py`:
   ```python
   from openjarvis.core.registry import MemoryRegistry
   from openjarvis.tools.storage._stubs import MemoryBackend, RetrievalResult

   @MemoryRegistry.register("my-backend")
   class MyBackend(MemoryBackend):
       def store(self, content, *, source="", metadata=None) -> str: ...
       def retrieve(self, query, top_k=5) -> list[RetrievalResult]: ...
       def delete(self, doc_id) -> bool: ...
       def clear(self) -> None: ...
   ```
2. Import in `tools/storage/__init__.py` with try/except for optional deps
3. Add test file with `pytest.importorskip()` if using optional deps
4. Add optional dep group in `pyproject.toml` if needed

### New Agent

1. Create `src/openjarvis/agents/my_agent.py`:
   ```python
   from openjarvis.agents._stubs import AgentResult, BaseAgent
   from openjarvis.core.registry import AgentRegistry

   @AgentRegistry.register("my-agent")
   class MyAgent(BaseAgent):
       agent_id = "my-agent"
       def __init__(self, engine, model, *, bus=None, **kwargs): ...
       def run(self, input, context=None, **kwargs) -> AgentResult: ...
   ```
2. Import in `agents/__init__.py`
3. Add test file `tests/agents/test_my_agent.py`

### New Engine

1. Create `src/openjarvis/engine/my_engine.py`:
   ```python
   from openjarvis.core.registry import EngineRegistry
   from openjarvis.engine._stubs import InferenceEngine

   @EngineRegistry.register("my-engine")
   class MyEngine(InferenceEngine):
       engine_id = "my-engine"
       def generate(self, messages, *, model, **kwargs) -> dict: ...
       def stream(self, messages, *, model, **kwargs): ...
       def list_models(self) -> list[str]: ...
       def health(self) -> bool: ...
   ```
2. Import in `engine/__init__.py` with try/except
3. Add to `_discovery.py` engine priority list if auto-discoverable

---

## Session Log

### Session 1 (2026-02-16) — Phase 5 Implementation

**Scope:** Full Phase 5 (v1.0) — SDK, OpenClaw, Benchmarks, Docker, Docs

**Work completed:**
- Step 1: Added `BenchmarkRegistry` to `core/registry.py`, updated `conftest.py`
- Step 2: Created `bench/` package — `_stubs.py`, `latency.py`, `throughput.py`, `__init__.py`; CLI `bench_cmd.py`
- Step 3: Created `sdk.py` — `Jarvis` class + `MemoryHandle`; updated `__init__.py` exports
- Step 4: Created OpenClaw infra — `openclaw_protocol.py`, `openclaw_transport.py`, `openclaw_plugin.py`; rewrote `openclaw.py` from stub
- Step 5: Created `Dockerfile`, `Dockerfile.gpu`, `docker-compose.yml`, `deploy/systemd/openjarvis.service`, `deploy/launchd/com.openjarvis.plist`
- Step 6: Version bump to 1.0.0, updated `README.md`, `CLAUDE.md`

**Bugs fixed during implementation:**
1. Ruff lint: 17 issues (E501, I001, F401, F841) — all fixed
2. Registry clearing broke `@register` decorators — switched to `ensure_registered()` for bench modules
3. `ChunkConfig(overlap=...)` should be `ChunkConfig(chunk_overlap=...)` — fixed
4. `chunk.text` should be `chunk.content` — fixed
5. Test content too short for chunking (0 chunks produced) — used 100 words

**Final: 520 passed, 8 skipped, 0 failures, ruff clean**

### Session 2 (2026-02-17) — Test Fixes + Live vLLM Testing

**Scope:** Fix broken tests, set up live vLLM inference testing

**Work completed:**
- Fixed 6 failed + 13 errored tests in `tests/cli/test_ask_router.py` and `tests/cli/test_ask_agent.py`
  - **Root cause:** `from openjarvis.cli.ask import ask` in `cli/__init__.py` shadows the `ask` module with the Click command object. When `mock.patch("openjarvis.cli.ask.get_engine")` resolves, it tries to patch an attribute on the Click command, not the module.
  - **Fix:** Use `importlib.import_module("openjarvis.cli.ask")` + `mock.patch.object(_ask_mod, "get_engine")` instead of string-based patching.
- Added tool fallback in `_openai_compat.py`: if server returns 400 when tools are sent (e.g., vLLM without `--enable-auto-tool-choice`), retry without tools.
- Verified live vLLM testing: existing vLLM server on port 8003 with `Qwen/Qwen3-8B`
- Tested: `jarvis ask`, `jarvis bench run`, `jarvis model list`, `jarvis memory index/search`, `jarvis telemetry stats`, SDK `Jarvis.ask()` and `ask_full()`

**Gotcha discovered:**
8. **Module shadowing with `from X import Y`** — If a package's `__init__.py` does `from openjarvis.cli.ask import ask`, then `openjarvis.cli.ask` in `sys.modules` is the *module*, but accessing it via attribute lookup on `openjarvis.cli` gives the imported *object* (the Click command). Use `importlib.import_module()` for reliable module access when patching.

**Live vLLM setup notes:**
- vLLM 0.15.1 running on Lambda cluster (8x A100-SXM4-80GB)
- Config: `~/.openjarvis/config.toml` with `vllm_host = "http://localhost:8003"` and `default_model = "Qwen/Qwen3-8B"`
- Tool calling requires `--enable-auto-tool-choice --tool-call-parser hermes` flags on vLLM server
- Without tool support, orchestrator falls back to reasoning-only mode

**Final: 520 passed, 8 skipped, 0 failures, ruff clean**

### Session 3 (2026-02-21) — Trace System & Research Direction

**Scope:** Design new research direction (abstractions for local AI), implement trace system

**Design decisions made:**
- OpenJarvis repositioned as a research framework for studying on-device AI
- Four core abstractions: Intelligence, Engine, Agentic Logic, Memory
- Learning is a cross-cutting concern driven by interaction traces
- Agentic Logic should be pluggable — users bring their own architecture (ReAct, OpenHands-style, etc.)
- Trace collection is the bridge between static and learned agents
- Evolve existing codebase rather than full redesign
- Name stays as OpenJarvis
- Learning focus: telemetry-driven routing/tool policies (lightweight, always-on)
- Agent-model coupling: loose (any agent, any model)

**Work completed:**
- Added `StepType` enum, `TraceStep`, `Trace` dataclasses to `core/types.py`
- Added `TRACE_STEP`, `TRACE_COMPLETE` event types to `core/events.py`
- Created `traces/` package:
  - `store.py` — `TraceStore`: SQLite-backed, save/get/list with filters, event bus subscription
  - `collector.py` — `TraceCollector`: wraps any `BaseAgent`, subscribes to EventBus, records steps automatically
  - `analyzer.py` — `TraceAnalyzer`: per-route stats, per-tool stats, summaries, export, query-type filtering
- Created `learning/trace_policy.py` — `TraceDrivenPolicy`: learns routing from trace outcomes, batch/online updates, registered as `"learned"` policy
- Registered `TraceDrivenPolicy` in `learning/__init__.py`
- Added 56 new tests across 4 test files in `tests/traces/` and `tests/learning/test_trace_policy.py`
- Updated all markdown documentation (README, VISION, ROADMAP, NOTES, CLAUDE)

**Final: 576 passed, 8 skipped, 0 failures, ruff clean**

### Session 4 (2026-03-02) — Codebase Simplification

**Scope:** Structural cleanup — reduce root sprawl, move data into package, remove shims, consolidate duplicate code

**PR:** [#2](https://github.com/HazyResearch/OpenJarvis/pull/2) — merged to main, 13 commits, 180 files changed

**Root directory:** 32 items → 20 items

**Key changes:**

1. **Removed generated artifacts** — deleted `get-pip.py` (2.2MB), added to `.gitignore`

2. **Docker files → `deploy/docker/`** — moved `Dockerfile`, `Dockerfile.gpu`, `Dockerfile.gpu.rocm`, `Dockerfile.sandbox`, `docker-compose.yml`, `docker-compose.gpu.rocm.yml`. Updated build contexts in compose files from `.` to `../..`

3. **Data files → package data** — moved TOML configs into `src/openjarvis/*/data/` directories:
   - `recipes/*.toml` → `src/openjarvis/recipes/data/`
   - `templates/agents/*.toml` → `src/openjarvis/templates/data/`
   - `skills/builtin/*.toml` → `src/openjarvis/skills/data/`
   - `operators/*.toml` → `src/openjarvis/operators/data/`
   - Discovery uses `Path(__file__).resolve().parent / "data"` pattern
   - Hatchling auto-includes non-Python files in the package tree (no explicit config needed)

4. **Evals → `src/openjarvis/evals/`** — moved entire `evals/` directory (69 files) into the package. All `from evals.` imports rewritten to `from openjarvis.evals.`. Updated `pyproject.toml` ruff per-file-ignores paths. Fixed stale `patch("evals.cli._run_single")` in test_config.py and stale `sys.path.insert` in conftest.py.

5. **Removed `memory/` backward-compat shims** — deleted 11 files that were pure re-exports from `tools/storage/`. Updated all imports across `cli/`, `sdk.py`, `system.py`, and 10+ test files.

6. **Consolidated 5 engine wrappers** — replaced `engine/vllm.py`, `sglang.py`, `llamacpp.py`, `mlx.py`, `lmstudio.py` (near-identical OpenAI-compat wrappers) with single data-driven `engine/openai_compat_engines.py` using `type()` for dynamic class creation from a config dict.

7. **Refactored `channels/__init__.py`** — replaced 26 identical try/except import blocks with `importlib.import_module()` loop (147 → 54 lines).

8. **Updated `CLAUDE.md`** — all path references updated for new structure.

**Stats:**
- Net code: −274 lines
- Lint errors: 178 → 117 (all remaining are pre-existing in untouched files)
- Tests: 3270 passed, 44 skipped, 0 failures

**Gotchas encountered:**
- `patch()` string targets inside test files aren't caught by `sed` import rewrites — found by spec review
- Subagent git commits may not persist to main working tree — verify with `git log` after subagent completes
- `/home/ubuntu/.local/bin/gh` (v0.0.4) shadows `/usr/bin/gh` (v2.4.0) — use full path for GitHub CLI

### Session 5 (2026-03-02) — Phase 23: Differentiated Functionalities

**Scope:** Make OpenJarvis's capabilities genuinely differentiated through trace-driven learning, real benchmarks, composable recipes, and operator recipes.

**PR:** Merged to main via `feat/phase-23-differentiated-functionalities` branch (squashed commit)

**Design:** "Learning Flywheel" approach — traces feed a learning pipeline that mines SFT pairs, fine-tunes models via LoRA, and evolves agent configs. Combined with 15 real academic benchmarks from IPW, composable recipes, and persistent operator agents.

**Implementation plan:** 15 tasks in 5 sections, executed via subagent-driven development with parallel dispatch.

**Work completed:**

1. **Trace-Driven Learning Pipeline (Tasks 1-4):**
   - `TrainingDataMiner` (`learning/training/data.py`) — extracts SFT, routing, and agent behavior pairs from traces with quality filters (min steps, success-only, dedup)
   - `LoRATrainer` (`learning/training/lora.py`) — LoRA fine-tuning with configurable rank/alpha/lr, evaluation loop, checkpoint management (requires torch)
   - `AgentConfigEvolver` (`learning/agent_evolver.py`) — LM-guided analysis of trace patterns to recommend agent config changes (tools, temperature, max_turns, system prompt)
   - `LearningOrchestrator` (`learning/learning_orchestrator.py`) — coordinates mine→train→evolve cycle on schedule, wired into `SystemBuilder` with lazy import and graceful degradation

2. **Eval Framework — 15 Real IPW Benchmarks (Tasks 5-7):**
   - Removed toy datasets (MT-Bench, HumanEval, RAG) per user's explicit request
   - Ported 11 new datasets + scorers from IPW benchmark interface to OpenJarvis `EvalRecord` interface
   - Key adaptation: IPW loads eagerly in `__init__`, OpenJarvis defers to `load()` method; `DatasetRecord.answer` → `EvalRecord.reference`
   - MCQ scorers (GPQA, MMLU-Pro): LLM letter extraction
   - Reasoning scorers (MATH-500, Natural Reasoning, HLE): exact match + `\boxed{}` extraction + LLM fallback
   - Agentic scorers (SWE-bench, SWEfficiency, TerminalBench Native): structural validation
   - CLI: `jarvis eval list|run|compare|report`

3. **Composable Abstractions (Tasks 8-10):**
   - Recipe system (`recipes/loader.py`) — `Recipe` dataclass with `to_builder_kwargs()`, `load_recipe()`, `discover_recipes()`, `resolve_recipe()`
   - 3 built-in recipes: `coding_assistant` (native_react, temp 0.3), `research_assistant` (orchestrator, 15 turns), `general_assistant` (orchestrator, temp 0.7)
   - 15 agent templates in TOML: code-reviewer, debugger, architect, deep-researcher, fact-checker, summarizer, inbox-triager, meeting-prep, note-taker, assistant, tutor, translator, writer, data-analyst, security-auditor
   - 20 bundled skills in TOML across 5 categories: file management, research, code quality, productivity, document processing

4. **Operator Recipes (Tasks 11-13):**
   - Researcher operator — 4-hour cron cycle, 8 tools, 20 max turns, deep research with memory indexing
   - Correspondent operator — 5-minute interval, 4 tools, 15 max turns, message triage
   - Sentinel operator — 2-hour cron cycle, 6 tools, 15 max turns, security/health monitoring

5. **Integration & Wiring (Tasks 14-15):**
   - Added 6 training config fields to `LearningConfig`: `training_enabled`, `training_schedule`, `lora_rank`, `lora_alpha`, `min_sft_pairs`, `min_improvement`
   - Wired `LearningOrchestrator` into `SystemBuilder._setup_learning_orchestrator()`
   - Updated CLAUDE.md for Phase 23

**Files:** 102 changed, ~11,500 lines added

**Merge notes:**
- Created feature branch from `origin/main`, squashed 19 commits into 1
- One merge conflict in `src/openjarvis/cli/__init__.py` (both PR #1 and this PR added new CLI commands) — resolved by keeping both `quickstart` and `eval_group`/`operators`

**Final: 3270 passed, 44 skipped, 0 failures**
