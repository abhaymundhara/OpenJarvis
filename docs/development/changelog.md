# Changelog

All notable changes to OpenJarvis are documented in this file.

---

## Unreleased — Phase 11 (NanoClaw Subsumption)

*27 new files, ~3,565 lines, 147+ new tests. Full suite: 2059+ tests pass.*

### Added

- **`ClaudeCodeAgent`** (`agents/claude_code.py`) -- Wraps the
  `@anthropic-ai/claude-code` SDK via a bundled Node.js subprocess bridge.
  Communicates over stdin/stdout using sentinel-delimited JSON
  (`---OPENJARVIS_OUTPUT_START---` / `---OPENJARVIS_OUTPUT_END---`). The
  bundled runner is auto-installed to `~/.openjarvis/claude_code_runner/` via
  `npm install --production` on first use. Registered as `"claude_code"` with
  `accepts_tools = False`. Requires Node.js 22+ and `ANTHROPIC_API_KEY`.
- **`WhatsAppBaileysChannel`** (`channels/whatsapp_baileys.py`) -- Bidirectional
  WhatsApp messaging using the Baileys protocol. Spawns a Node.js bridge
  subprocess (`whatsapp_baileys_bridge/`) for QR-code authentication, incoming
  message forwarding, and outbound delivery via JID addressing. Registered as
  `"whatsapp_baileys"` in `ChannelRegistry`. Authentication state is persisted
  to `~/.openjarvis/whatsapp_baileys_bridge/auth/`. New config section:
  `[channel.whatsapp_baileys]`.
- **`ContainerRunner`** (`sandbox/runner.py`) -- Manages Docker (or Podman)
  container lifecycle for sandboxed agent execution. Builds `docker run --rm
  --network none -i` commands with allowlist-validated read-only bind mounts.
  Supports configurable image, timeout, concurrent container limit, and runtime
  binary. Uses the same sentinel-delimited JSON protocol as `ClaudeCodeAgent`.
- **`SandboxedAgent`** (`sandbox/runner.py`) -- Transparent wrapper that runs
  any `BaseAgent` inside a container via `ContainerRunner`. Follows the
  `GuardrailsEngine` wrapper pattern. `accepts_tools = False`.
- **`MountAllowlist` / `validate_mounts()`** (`sandbox/mount_security.py`) --
  Port of NanoClaw's `mount-security.ts`. Validates bind mounts against a JSON
  allowlist (allowed root directories + blocked filename patterns). Raises
  `ValueError` for blocked or out-of-root paths before the container starts.
  Default blocked patterns include `.ssh`, `.env`, `*.pem`, `*.key`, credential
  files, and cloud config directories.
- **`TaskScheduler`** (`scheduler/scheduler.py`) -- Background polling scheduler
  supporting three schedule types: `cron` (via `croniter` or built-in fallback),
  `interval` (seconds), and `once` (ISO 8601 datetime). Runs a daemon thread
  (`jarvis-scheduler`) polling SQLite every 60 seconds (configurable). Executes
  due tasks via `JarvisSystem.ask()` with optional agent and tool selection.
  Publishes `scheduler_task_start` / `scheduler_task_end` events on the
  `EventBus`. New config section: `[scheduler]`.
- **`SchedulerStore`** (`scheduler/store.py`) -- SQLite CRUD backend for
  scheduled tasks and run logs. Two tables: `scheduled_tasks` (task state) and
  `task_run_logs` (execution history). Supports task filtering by status and
  due-time polling via `get_due_tasks()`.
- **Scheduler MCP tools** (`scheduler/tools.py`) -- Five new MCP-discoverable
  tools registered in `ToolRegistry`:
    - `schedule_task` -- Create a new scheduled task
    - `list_scheduled_tasks` -- List tasks filtered by status
    - `pause_scheduled_task` -- Pause an active task
    - `resume_scheduled_task` -- Resume a paused task (recomputes `next_run`)
    - `cancel_scheduled_task` -- Permanently cancel a task
- **Scheduler CLI commands** -- `jarvis scheduler` subcommand group:
    - `jarvis scheduler create` -- Create a new scheduled task
    - `jarvis scheduler list` -- List all or filtered tasks
    - `jarvis scheduler pause <id>` -- Pause a task
    - `jarvis scheduler resume <id>` -- Resume a task
    - `jarvis scheduler cancel <id>` -- Cancel a task
    - `jarvis scheduler logs <id>` -- Show run history for a task
    - `jarvis scheduler start` -- Start the background scheduler daemon

### Changed

- `ChannelRegistry` now includes `WhatsAppBaileysChannel`.
- `AgentRegistry` now includes `ClaudeCodeAgent` (`"claude_code"`).
- Architecture overview and source directory layout updated to reflect new
  `sandbox/` and `scheduler/` modules.

---

## Unreleased — Phase 10 Tooling Updates

### Added

- **`build_tool_descriptions()` shared builder** -- Single source of truth for
  generating enriched tool descriptions in agent system prompts. Produces
  Markdown sections with name, description, category, and parameter schemas.
- **Enriched agent prompts** -- `NativeReActAgent`, `NativeOpenHandsAgent`,
  `RLMAgent`, and `OrchestratorAgent` (structured mode) now inject detailed
  tool descriptions into their system prompts via the shared builder.
- **Case-insensitive parsing** -- ReAct (`Action:` / `Final Answer:`) and
  Orchestrator structured-mode parsing (`TOOL:` / `FINAL_ANSWER:`) are now
  case-insensitive.
- **Multi-provider tool_calls extraction** -- `CloudEngine` now extracts
  `tool_calls` from Anthropic (`tool_use` content blocks) and Google
  (`function_call` parts), normalizing to the flat `{id, name, arguments}`
  format. `LiteLLM` engine handles the flat-format tool calls returned by
  the LiteLLM proxy.
- **RLM tool awareness** -- `RLMAgent` injects an `## Available Tools`
  section into its system prompt when tools are provided.
- **Orchestrator structured tool descriptions** -- Structured mode passes
  `tools=self._tools` to `build_system_prompt()` for enriched descriptions.
- **Telemetry modules** -- `EfficiencyMetrics`, `GPUMonitor`, `VLLMMetrics`
  for energy, GPU utilization, and vLLM server-side metrics collection.
- **Eval TOML config** -- TOML-based eval suite configuration system for
  defining models x benchmarks matrices.

### Changed

- Agent prompt generation now uses `build_tool_descriptions()` instead of
  inline tool name listing.
- `build_system_prompt()` in `prompt_registry.py` accepts an optional `tools`
  parameter for enriched descriptions from `BaseTool` instances.
- ReAct and OpenHands regex patterns updated for case-insensitive matching.

### Fixed

- Engine `tool_calls` normalization -- Anthropic `tool_use` blocks and Google
  `function_call` parts are now correctly extracted and converted to the
  standard flat format used by agents.

---

## v0.1.0

*Phase 5 -- SDK, Production Readiness, and Documentation*

### Added

- **Python SDK** -- `Jarvis` class providing a high-level sync API for
  programmatic use
    - `ask()` / `ask_full()` methods for direct engine and agent mode queries
    - `MemoryHandle` proxy for lazy memory backend initialization
    - `list_models()` and `list_engines()` for runtime introspection
    - Router policy selection via config (`learning.default_policy`)
    - Lazy engine initialization with automatic discovery and health probing
    - Resource cleanup via `close()`
- **Benchmarking framework**
    - `BaseBenchmark` ABC and `BenchmarkSuite` runner
    - `LatencyBenchmark` measuring per-call latency (mean, p50, p95, min, max)
    - `ThroughputBenchmark` measuring tokens-per-second throughput
    - `BenchmarkResult` dataclass with JSONL export
    - `jarvis bench run` CLI with options for model, engine, sample count,
      benchmark selection, and JSON/JSONL output
- **Docker deployment**
    - `Dockerfile` -- Multi-stage Python 3.12-slim build with `[server]` extra
    - `Dockerfile.gpu` -- NVIDIA CUDA 12.4 runtime variant
    - `docker-compose.yml` -- Services for `jarvis` (port 8000) and `ollama`
      (port 11434)
    - `deploy/systemd/openjarvis.service` -- systemd unit file for Linux
    - `deploy/launchd/com.openjarvis.plist` -- launchd plist for macOS
- **Documentation site** -- MkDocs Material with mkdocstrings, covering
  getting started, user guide, architecture, API reference, deployment, and
  development

---

## v0.5.0

*Phase 4 -- Learning, Telemetry, and Router Policies*

### Added

- **Learning system**
    - `RouterPolicy` ABC and `RoutingContext` dataclass
    - `RewardFunction` ABC for scoring inference results
    - `HeuristicRewardFunction` scoring on latency, cost, and efficiency
    - `RouterPolicyRegistry` for pluggable routing strategies
    - `HeuristicRouter` registered as `"heuristic"` policy (6 priority rules:
      code detection, math detection, short/long queries, urgency override,
      default fallback)
    - `TraceDrivenPolicy` registered as `"learned"` policy with batch updates
      via `update_from_traces()` and online updates via `observe()`
    - `GRPORouterPolicy` stub registered as `"grpo"` for future RL training
    - `ensure_registered()` pattern for lazy, test-safe registration
- **Telemetry aggregation**
    - `TelemetryAggregator` with `per_model_stats()`, `per_engine_stats()`,
      `top_models()`, `summary()`, `export_records()`, and `clear()` methods
    - Time-range filtering via `since` / `until` parameters
    - `ModelStats` and `EngineStats` dataclasses
    - `AggregatedStats` summary dataclass
- **CLI enhancements**
    - `--router` flag on `jarvis ask` for explicit policy selection
    - `jarvis telemetry stats` -- display aggregated telemetry statistics
    - `jarvis telemetry export --format json|csv` -- export telemetry records
    - `jarvis telemetry clear --yes` -- delete all telemetry records

---

## v0.4.0

*Phase 3 -- Agents, Tools, and API Server*

### Added

- **Agent system**
    - `BaseAgent` ABC with `run()` method returning `AgentResult`
    - `AgentContext` dataclass with conversation, tools, and memory results
    - `AgentResult` dataclass with content, tool results, turns, and metadata
    - `AgentRegistry` for pluggable agent implementations
    - `SimpleAgent` -- single-turn query-to-response, no tool calling
    - `OrchestratorAgent` -- multi-turn tool-calling loop with `ToolExecutor`,
      configurable `max_turns`
    - `CustomAgent` -- template for user-defined agent behavior
- **Tool system**
    - `BaseTool` ABC with `spec` property and `execute()` method
    - `ToolSpec` dataclass describing tool interface and characteristics
    - `ToolExecutor` dispatch engine with JSON argument parsing, latency
      tracking, and event bus integration (`TOOL_CALL_START` / `TOOL_CALL_END`)
    - `ToolRegistry` for tool discovery
    - `to_openai_function()` method for OpenAI function calling format
    - Built-in tools:
        - `CalculatorTool` -- safe math evaluation via AST parsing
        - `ThinkTool` -- reasoning scratchpad for chain-of-thought
        - `RetrievalTool` -- memory search integration
        - `LLMTool` -- sub-model calls within agent loops
        - `FileReadTool` -- safe file reading with path validation
- **OpenAI-compatible API server** (`jarvis serve`)
    - FastAPI + Uvicorn with optional `[server]` extra
    - `POST /v1/chat/completions` -- non-streaming and SSE streaming
    - `GET /v1/models` -- list available models
    - `GET /health` -- health check endpoint
    - Pydantic request/response models matching OpenAI API format

---

## v0.3.0

*Phase 2 -- Memory System*

### Added

- **Memory backends**
    - `MemoryBackend` ABC with `store()`, `retrieve()`, `delete()`, `clear()`
    - `RetrievalResult` dataclass with content, score, source, and metadata
    - `MemoryRegistry` for backend discovery
    - `SQLiteMemory` -- zero-dependency default using SQLite FTS5 with BM25
      ranking and FTS5 query escaping
    - `FAISSMemory` -- vector search using FAISS with sentence-transformers
      embeddings (optional `[memory-faiss]` extra)
    - `ColBERTMemory` -- ColBERTv2 neural retrieval backend (optional
      `[memory-colbert]` extra)
    - `BM25Memory` -- BM25 ranking backend using rank-bm25 (optional
      `[memory-bm25]` extra)
    - `HybridMemory` -- Reciprocal Rank Fusion combining multiple backends
- **Document processing**
    - `ChunkConfig` dataclass for chunk size and overlap settings
    - `chunk_text()` for splitting documents into overlapping chunks
    - `ingest_path()` for recursively indexing files and directories
    - `read_document()` with support for plain text, Markdown, and PDF
      (optional `[memory-pdf]` extra)
- **Context injection**
    - `ContextConfig` with top-k, minimum score, and max context token settings
    - `inject_context()` for prepending memory results as system messages with
      source attribution
    - `--no-context` flag on `jarvis ask` to disable injection
- **CLI commands**
    - `jarvis memory index <path>` -- index documents into memory
    - `jarvis memory search <query>` -- search memory for relevant chunks
    - `jarvis memory stats` -- show backend statistics
- **Event bus integration** -- `MEMORY_STORE` and `MEMORY_RETRIEVE` events

---

## v0.2.0

*Phase 1 -- Intelligence and Inference*

### Added

- **Intelligence primitive**
    - `ModelSpec` dataclass with parameter count, context length, quantization,
      VRAM requirements, and supported engines
    - `ModelRegistry` for model metadata storage
    - `BUILTIN_MODELS` catalog with pre-defined model specifications
    - `register_builtin_models()` and `merge_discovered_models()` helpers
    - `HeuristicRouter` with rule-based model selection
    - `build_routing_context()` for query analysis (code detection, math
      detection, length classification)
- **Inference engines**
    - `InferenceEngine` ABC with `generate()`, `stream()`, `list_models()`,
      and `health()` methods
    - `EngineRegistry` for engine discovery
    - `OllamaEngine` -- Ollama backend via native HTTP API with tool call
      extraction
    - `VllmEngine` -- vLLM backend via OpenAI-compatible API
    - `LlamaCppEngine` -- llama.cpp server backend
    - `EngineConnectionError` for unreachable engines
    - `messages_to_dicts()` for Message-to-OpenAI-format conversion
- **Engine discovery**
    - `discover_engines()` -- probe all registered engines for health
    - `discover_models()` -- aggregate model lists across engines
    - `get_engine()` -- get configured default with automatic fallback
- **Hardware detection**
    - NVIDIA GPU detection via `nvidia-smi`
    - AMD GPU detection via `rocm-smi`
    - Apple Silicon detection via `system_profiler`
    - CPU brand detection via `/proc/cpuinfo` and `sysctl`
    - `recommend_engine()` mapping hardware to best engine
- **Telemetry**
    - `TelemetryRecord` dataclass with timing, tokens, energy, and cost
    - `TelemetryStore` with SQLite persistence and EventBus subscription
    - `instrumented_generate()` wrapper for automatic telemetry recording
- **CLI**
    - `jarvis ask <query>` -- query via discovered engine
    - `jarvis ask --agent simple <query>` -- route through SimpleAgent
    - `jarvis model list` -- list models from running engines
    - `jarvis model info <model>` -- show model details

---

## v0.1.0

*Phase 0 -- Project Scaffolding*

### Added

- **Project structure** -- `hatchling` build backend, `uv` package manager,
  `pyproject.toml` with extras for optional backends
- **Registry system** -- `RegistryBase[T]` generic base class with
  class-specific entry isolation, `register()` decorator, `get()`, `create()`,
  `items()`, `keys()`, `contains()`, `clear()` methods
- **Typed registries** -- `ModelRegistry`, `EngineRegistry`, `MemoryRegistry`,
  `AgentRegistry`, `ToolRegistry`, `RouterPolicyRegistry`, `BenchmarkRegistry`
- **Core types** -- `Role` enum, `Message`, `Conversation` (with sliding
  window), `ModelSpec`, `Quantization` enum, `ToolCall`, `ToolResult`,
  `TelemetryRecord`, `StepType` enum, `TraceStep`, `Trace`
- **Configuration** -- `JarvisConfig` dataclass hierarchy, TOML loader with
  overlay semantics, hardware auto-detection, `generate_default_toml()` for
  `jarvis init`
- **Event bus** -- Synchronous pub/sub `EventBus` with `EventType` enum for
  inter-primitive communication
- **CLI skeleton** -- Click-based `jarvis` command group with `--version`,
  `--help`, and `init` subcommand
