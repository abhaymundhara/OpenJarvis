# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Status

OpenJarvis is a research framework for studying on-device AI systems. Phase 25 complete. Five composable pillars: Intelligence, Engine, Agents, Tools (with storage + MCP), and Learning — with trace-driven learning as a cross-cutting concern. Speech subsystem (STT) with pluggable backends. ~3405 tests pass (~44 skipped for optional deps). Python SDK (`Jarvis` class), composition layer (`SystemBuilder`/`JarvisSystem`), eval framework (20 real benchmarks), composable recipes, agent templates, bundled skills, operator recipes, trace-driven learning pipeline, Docker deployment, Tauri desktop app, 41+ tools, 20+ CLI commands, 40+ API endpoints all ready.

## Build & Development Commands

```bash
uv sync --extra dev          # Install deps + dev tools
uv run pytest tests/ -v      # Run ~3295 tests (~44 skipped if optional deps missing)
uv run ruff check src/ tests/ # Lint
uv run jarvis --version      # 1.0.0
uv run jarvis ask "Hello"    # Query via discovered engine (direct mode)
uv run jarvis ask --agent simple "Hello"           # SimpleAgent route
uv run jarvis ask --agent orchestrator "Hello"     # OrchestratorAgent route
uv run jarvis ask --agent orchestrator --tools calculator,think "What is 2+2?"
uv run jarvis ask --agent native_react --tools calculator "What is 2+2?"  # NativeReActAgent
uv run jarvis ask --agent react "Hello"            # Alias for native_react
uv run jarvis ask --agent native_openhands "Hello" # NativeOpenHandsAgent (CodeAct)
uv run jarvis ask --agent openhands "Hello"        # Real OpenHands SDK (requires openhands-sdk)
uv run jarvis ask --router heuristic "Hello"       # Explicit heuristic policy
uv run jarvis ask --no-context "Hello"  # Query without memory context injection
uv run jarvis model list     # List models from running engines
uv run jarvis model info qwen3:8b  # Show model details
uv run jarvis memory index ./docs/   # Index documents into memory
uv run jarvis memory search "topic"  # Search memory for relevant chunks
uv run jarvis memory stats           # Show memory backend statistics
uv run jarvis telemetry stats        # Show aggregated telemetry stats
uv run jarvis telemetry export --format json  # Export records as JSON
uv run jarvis telemetry export --format csv   # Export records as CSV
uv run jarvis telemetry clear --yes  # Delete all telemetry records
uv run jarvis channel list           # List available messaging channels
uv run jarvis channel send slack "Hello"  # Send a message to a channel
uv run jarvis channel status         # Show channel bridge connection status
uv run jarvis scheduler create "Check weather" --type cron --value "0 9 * * *"
uv run jarvis scheduler list         # List scheduled tasks
uv run jarvis scheduler start        # Start scheduler daemon (foreground)
uv run jarvis bench run              # Run all benchmarks against engine
uv run jarvis bench run -b energy -w 5 -n 20 --json  # Energy benchmark with warmup
uv run jarvis serve --port 8000      # OpenAI-compatible API server (requires openjarvis[server])
uv run jarvis doctor                 # Run diagnostic checks (config, engines, models, deps)
uv run jarvis doctor --json          # Machine-readable diagnostics
uv run jarvis start                  # Start server as background daemon
uv run jarvis stop                   # Stop background daemon
uv run jarvis restart                # Restart background daemon
uv run jarvis status                 # Show daemon status (PID, uptime)
uv run jarvis chat                   # Interactive REPL (/quit, /clear, /model, /help, /history)
uv run jarvis chat --agent orchestrator --tools calculator  # REPL with agent
uv run jarvis agent list             # List registered agents
uv run jarvis agent info native_react  # Show agent details
uv run jarvis workflow list          # List available workflows
uv run jarvis workflow run my_workflow  # Execute a workflow
uv run jarvis skill list             # List installed skills
uv run jarvis skill install path/to/skill.toml  # Install a skill
uv run jarvis vault set MY_KEY       # Store encrypted credential
uv run jarvis vault get MY_KEY       # Retrieve credential
uv run jarvis vault list             # List stored keys
uv run jarvis add github             # Quick-add MCP server (github, slack, postgres, etc.)
uv run jarvis eval list                            # List 15 benchmarks and backends
uv run jarvis eval run -b supergpqa -m "qwen3:8b"  # Single benchmark run
uv run jarvis eval run -c src/openjarvis/evals/configs/suite.toml  # Suite mode (models x benchmarks)
uv run jarvis eval compare result1.jsonl result2.jsonl  # Compare runs side-by-side
uv run jarvis eval report result.jsonl              # Detailed report with per-subject breakdown
uv run jarvis --help         # Show all subcommands
uv run jarvis init --force   # Detect hardware, write ~/.openjarvis/config.toml
# Eval framework (direct module invocation)
source .env                  # Load API keys before running evals
uv run python -m openjarvis.evals run -c src/openjarvis/evals/configs/glm-4.7-flash-openhands.toml -v  # Run eval suite from TOML config
uv run python -m openjarvis.evals run -b supergpqa -m "qwen3:8b" -n 50      # Run single benchmark
uv run python -m openjarvis.evals summarize results/supergpqa_qwen3-8b.jsonl # Summarize results
```

### Config File Conventions

- **Runtime config (source of truth):** `configs/openjarvis/config.toml` — Pillar-aligned OpenJarvis config. Copied to `~/.openjarvis/config.toml` at runtime (which is where `load_config()` reads from).
- **Eval suite configs:** `src/openjarvis/evals/configs/*.toml` — TOML configs defining models x benchmarks matrices.
- **API keys:** `.env` file in project root (gitignored). Source with `source .env` before running evals or cloud operations.
- **Never save configs to `~/.openjarvis/` directly** — always maintain the canonical copy in `configs/openjarvis/` and copy/symlink to `~/.openjarvis/`.

### Python SDK

```python
from openjarvis import Jarvis

j = Jarvis()                          # Uses default config + auto-detected engine
j = Jarvis(model="qwen3:8b")         # Override model
j = Jarvis(engine_key="ollama")       # Override engine

response = j.ask("Hello")            # Returns string
full = j.ask_full("Hello")           # Returns dict with content, usage, model, engine
response = j.ask("Hello", agent="orchestrator", tools=["calculator"])

j.memory.index("./docs/")            # Index documents
results = j.memory.search("topic")   # Search memory
j.memory.stats()                     # Backend stats

j.list_models()                       # Available models
j.list_engines()                      # Registered engines
j.close()                             # Release resources
```

- **Package manager:** `uv` with `hatchling` build backend
- **Config:** `pyproject.toml` with extras for optional backends (e.g., `openjarvis[inference-vllm]`, `openjarvis[inference-mlx]`, `openjarvis[memory-colbert]`, `openjarvis[server]`, `openjarvis[openclaw]`, `openjarvis[energy-amd]`, `openjarvis[energy-apple]`, `openjarvis[energy-all]`, `openjarvis[security-signing]`, `openjarvis[sandbox-wasm]`, `openjarvis[dashboard]`, `openjarvis[browser]`, `openjarvis[media]`, `openjarvis[pdf]`, `openjarvis[channel-line]`, `openjarvis[channel-viber]`, `openjarvis[channel-reddit]`, `openjarvis[channel-mastodon]`, `openjarvis[channel-xmpp]`, `openjarvis[channel-rocketchat]`, `openjarvis[channel-zulip]`, `openjarvis[channel-twitch]`, `openjarvis[channel-nostr]`, `openjarvis[speech]`, `openjarvis[speech-deepgram]`)
- **CLI entry point:** `jarvis` (Click-based) — subcommands: `init`, `ask`, `serve`, `start`, `stop`, `restart`, `status`, `chat`, `model`, `memory`, `telemetry`, `bench`, `eval`, `channel`, `scheduler`, `doctor`, `agent`, `workflow`, `skill`, `vault`, `add`
- **Python:** 3.10+ required
- **Node.js:** 22+ required only for OpenClaw agent

## Architecture

OpenJarvis is a research framework for on-device AI organized around **five composable pillars**, each with a clear ABC interface and a decorator-based registry for runtime discovery.

### Five Pillars

1. **Intelligence** (`src/openjarvis/intelligence/`) — Model definition, catalog, and generation defaults. `ModelRegistry` maps model keys to `ModelSpec`. `IntelligenceConfig` holds model identity (default/fallback model, model_path, checkpoint_path, quantization, preferred_engine, provider) and generation defaults (temperature, max_tokens, top_p, top_k, repetition_penalty, stop_sequences). Model catalog maintains `BUILTIN_MODELS` with auto-discovery via `merge_discovered_models()`.
2. **Engine** (`src/openjarvis/engine/`) — The inference runtime. Backends: vLLM, SGLang, Ollama, llama.cpp, MLX, LM Studio. All implement `InferenceEngine` ABC with `generate()`, `stream()`, `list_models()`, `health()`. Engines extract and pass through `tool_calls` in OpenAI format.
3. **Agents** (`src/openjarvis/agents/`) — Pluggable logic for queries, tool/API calls, memory. Hierarchy: `BaseAgent` ABC (helpers: `_emit_turn_start/end`, `_build_messages`, `_generate`, `_max_turns_result`, `_strip_think_tags`, `_check_continuation`) → `ToolUsingAgent` (adds `tools`, `ToolExecutor`, `max_turns`). Agents: `SimpleAgent` (single-turn), `OrchestratorAgent` (multi-turn tool loop), `NativeReActAgent` (Thought-Action-Observation, key `"native_react"`, alias `"react"`), `NativeOpenHandsAgent` (CodeAct, key `"native_openhands"`), `RLMAgent` (recursive LM), `OpenHandsAgent` (real `openhands-sdk`, key `"openhands"`, requires Python 3.12+), `OpenClawAgent` (HTTP/subprocess transport), `ClaudeCodeAgent` (Claude Agent SDK via Node.js, key `"claude_code"`), `SandboxedAgent` (Docker wrapper, key `"sandboxed"`), `MonitorOperativeAgent` (long-horizon monitoring with 4 configurable strategies: memory extraction, observation compression, retrieval, task decomposition, key `"monitor_operative"`). `accepts_tools` class attribute for CLI/SDK auto-detection. Agents call `engine.generate()` directly — telemetry handled by `InstrumentedEngine` wrapper.
4. **Tools** (`src/openjarvis/tools/`) — All tools managed via MCP (Model Context Protocol).
   - **API tools**: `CalculatorTool`, `ThinkTool`, `FileReadTool`, `FileWriteTool`, `WebSearchTool`, `CodeInterpreterTool`, `LLMTool`, `ShellExecTool`, `ApplyPatchTool`, `HttpRequestTool`, `DatabaseQueryTool`, `PDFExtractTool`, `ImageGenerateTool`, `AudioTranscribeTool` — all implement `BaseTool` ABC
   - **Git tools** (`git_tool.py`): `GitStatusTool`, `GitDiffTool`, `GitCommitTool`, `GitLogTool`
   - **Browser tools** (`browser.py`): `BrowserNavigateTool`, `BrowserClickTool`, `BrowserTypeTool`, `BrowserScreenshotTool`, `BrowserExtractTool` (Playwright, optional `[browser]`)
   - **Agent tools** (`agent_tools.py`): `AgentSpawnTool`, `AgentSendTool`, `AgentListTool`, `AgentKillTool`
   - **Storage tools** (`storage_tools.py`): `MemoryStoreTool`, `MemoryRetrieveTool`, `MemorySearchTool`, `MemoryIndexTool`
   - **Storage backends** (`tools/storage/`): SQLite/FTS5 (default), FAISS, ColBERTv2, BM25, Hybrid (RRF fusion), KnowledgeGraph. All implement `MemoryBackend` ABC. Canonical import: `from openjarvis.tools.storage.sqlite import SQLiteMemory`.
   - **Scheduler tools** (`scheduler/tools.py`): 5 MCP tools for task scheduling
   - **Knowledge graph tools** (`knowledge_tools.py`): `KGAddEntityTool`, `KGAddRelationTool`, `KGQueryTool`, `KGNeighborsTool`
   - **MCP adapter** (`mcp_adapter.py`): `MCPToolAdapter` wraps external MCP tools as native `BaseTool`; `MCPToolProvider` discovers from server
   - **MCP server** (`mcp/server.py`): Exposes all built-in tools via JSON-RPC `tools/list` + `tools/call` (MCP spec 2025-11-25)
   - **MCP templates** (`tools/templates/`): `ToolTemplate` dynamically constructs tools from TOML specs. 10 builtin templates. `discover_templates()` auto-discovers.
   - **`ToolExecutor`**: dispatch with RBAC check + taint check, `timeout_seconds` on `ToolSpec` (default 30s via `ThreadPoolExecutor`), event bus integration
   - All registered via `@ToolRegistry.register("name")` decorator
5. **Learning** (`src/openjarvis/learning/`) — Structured learning with nested per-pillar sub-policies. `LearningConfig` sections: `routing` (heuristic/learned/grpo/bandit), `intelligence` (none/sft), `agent` (none/agent_advisor/icl_updater), `metrics` (accuracy/latency/cost/efficiency weights). Policies: `SFTRouterPolicy` (query→model from traces), `AgentAdvisorPolicy` (LM-guided), `ICLUpdaterPolicy` (in-context with example DB, versioning, rollback, quality gates), `GRPORouterPolicy` (softmax sampling, group relative advantage, per-query-class weights), `BanditRouterPolicy` (Thompson Sampling / UCB1, per-arm stats). `SkillDiscovery` mines tool subsequences from traces to auto-generate skill manifests. Router policies: `HeuristicRouter`, `TraceDrivenPolicy`. Orchestrator training subpackage provides SFT and GRPO pipelines. **Trace-driven learning pipeline**: `TrainingDataMiner` (extracts SFT pairs from traces with quality filters), `LoRATrainer` (LoRA fine-tuning with configurable rank/alpha, requires torch), `AgentConfigEvolver` (LM-guided agent config recommendations from trace patterns), `LearningOrchestrator` (wired into `SystemBuilder`, orchestrates mine→train→evolve cycle on schedule).

### Speech Subsystem

- **Speech** (`src/openjarvis/speech/`) — Speech-to-text with pluggable backends. `SpeechBackend` ABC (`transcribe()`, `health()`, `supported_formats()`). `TranscriptionResult` + `Segment` dataclasses. Backends: `FasterWhisperBackend` (local, CTranslate2, key `"faster-whisper"`), `OpenAIWhisperBackend` (cloud, `whisper-1`, key `"openai"`), `DeepgramSpeechBackend` (cloud, `nova-2`, key `"deepgram"`). Auto-discovery with local-first priority. `SpeechConfig` in `JarvisConfig`. `SpeechRegistry` for backend registration. Wired into `SystemBuilder.speech()`, `create_app()`, `jarvis serve`. API: `POST /v1/speech/transcribe` (multipart), `GET /v1/speech/health`. Frontend: `useSpeech` hook + `MicButton` component. Tauri: `transcribe_audio` + `speech_health` commands. Optional deps: `openjarvis[speech]` (faster-whisper), `openjarvis[speech-deepgram]` (deepgram-sdk).

### Cross-cutting Systems

- **Traces** (`src/openjarvis/traces/`) — Full interaction recording. `Trace` captures `TraceStep`s (route, retrieve, generate, tool_call, respond) with timing. `TraceStore` (SQLite), `TraceCollector` (auto-wraps agents), `TraceAnalyzer` (stats for learning).
- **Telemetry** (`src/openjarvis/telemetry/`) — `InstrumentedEngine` wraps any engine, publishing events to SQLite via `TelemetryStore`. `TelemetryAggregator` for read-only queries. `EnergyMonitor` ABC with vendor-specific implementations: `NvidiaEnergyMonitor` (hw counters/polling), `AmdEnergyMonitor` (amdsmi), `AppleEnergyMonitor` (zeus-ml), `RaplEnergyMonitor` (sysfs). `EnergyBatch` for batch-level energy-per-token. `SteadyStateDetector` for thermal equilibrium (CV-based).
- **Security** (`src/openjarvis/security/`) — `SecretScanner` + `PIIScanner` (implement `BaseScanner` ABC). `GuardrailsEngine` wraps engines with input/output scanning (WARN/REDACT/BLOCK modes). `AuditLogger` with Merkle hash chain (SHA-256 tamper-evidence). `CapabilityPolicy` RBAC (10 capabilities with glob matching, enforced in `ToolExecutor`). `TaintLabel`/`TaintSet` information flow control with `SINK_POLICY`. Ed25519 signing via `cryptography` (optional `[security-signing]`). `file_policy.py` for sensitive file detection. `InjectionScanner` (11 regex patterns: prompt override, identity override, code/shell injection, exfiltration, jailbreak, delimiter injection). `check_ssrf()` SSRF protection (RFC 1918, loopback, link-local, cloud metadata blocking). `RateLimiter` with `TokenBucket` (thread-safe, per-key). `run_sandboxed()` subprocess isolation (`os.setsid`, process group kill, env clearing). Security HTTP middleware (7 headers: CSP, HSTS, X-Frame-Options, etc.).

### Composition & Infrastructure

- **Composition Layer** (`system.py`) — `SystemBuilder` fluent builder → `JarvisSystem` with `ask()`, `close()`. Wires engine, model, agent, tools, telemetry, traces, workflow, sessions, capability policy.
- **SDK** (`sdk.py`) — `Jarvis` class: high-level sync API with `ask()`/`ask_full()`, `MemoryHandle`, lazy init, telemetry. Also exports `JarvisSystem`/`SystemBuilder`.
- **Benchmarks** (`bench/`) — `LatencyBenchmark`, `ThroughputBenchmark`, `EnergyBenchmark`. All registered via `BenchmarkRegistry`. CLI: `jarvis bench run`.
- **Eval Framework** (`src/openjarvis/evals/`) — 20 real benchmark datasets: SuperGPQA, GPQA, MMLU-Pro, MATH-500, Natural Reasoning, HLE, SimpleQA, WildChat, IPW, GAIA, FRAMES, SWE-bench, SWEfficiency, TerminalBench, TerminalBench Native, LogHub, AMA-Bench, LifelongAgentBench, WebChoreArena, WorkArena++. Scorer types: MCQ letter extraction, LLM-judge, exact match, structural validation. `EvalRunner` with parallel execution and episode mode (sequential processing with shared agent state). `EnvironmentProvider` ABC for Docker/ServiceNow environments. CLI: `jarvis eval list|run|compare|report`.
- **Recipes** (`src/openjarvis/recipes/`) — Composable TOML configs that wire all 5 pillars. `Recipe` dataclass with `to_builder_kwargs()`. `load_recipe()`, `discover_recipes()`, `resolve_recipe()`. 3 built-in recipes in `data/`: coding_assistant, research_assistant, general_assistant. Operator recipes in `data/operators/`: researcher (4h cycle), correspondent (5min interval), sentinel (2h cycle), monitor (2h cycle, causality graph + hybrid retrieval).
- **Agent Templates** (`src/openjarvis/templates/`) — Pre-configured TOML manifests with system prompts, tool sets, behavioral parameters. `AgentTemplate` dataclass, `load_template()`, `discover_templates()`. 15 built-in templates in `data/` (code-reviewer, debugger, architect, deep-researcher, fact-checker, summarizer, etc.).
- **Bundled Skills** (`src/openjarvis/skills/data/`) — 20 ready-to-use TOML skill manifests. Categories: file management (organizer, deduplicator, backup), research (web-summarize, topic-research, knowledge-extract), code quality (lint, test-gen, security-scan, dependency-audit), productivity (email-draft, meeting-notes, daily-digest), document processing (compare, translate, data-analyze).
- **OpenClaw** (`agents/openclaw*.py`) — `OpenClawAgent` with `HttpTransport`/`SubprocessTransport`, JSON-line protocol, `ProviderPlugin`, `MemorySearchManager`.
- **API Server** (`server/`) — OpenAI-compatible via `jarvis serve` (FastAPI + uvicorn). Endpoints: `POST /v1/chat/completions`, `GET /v1/models`, `GET /health`, channel endpoints. SSE streaming.
- **Channels** (`channels/`) — `BaseChannel` ABC. `OpenClawChannelBridge` (WebSocket/HTTP to OpenClaw gateway). `WhatsAppBaileysChannel` (Baileys protocol, Node.js bridge, QR auth). Phase 21 channels: `LINEChannel`, `ViberChannel`, `MessengerChannel`, `RedditChannel`, `MastodonChannel`, `XMPPChannel`, `RocketChatChannel`, `ZulipChannel`, `TwitchChannel`, `NostrChannel`. All follow `BaseChannel` ABC with env var fallbacks, `@ChannelRegistry.register()`, `EventBus` integration.
- **Sandbox** (`sandbox/`) — `ContainerRunner` (Docker/Podman lifecycle, mount validation). `WasmRunner` (wasmtime-py, fuel/memory limits, optional `[sandbox-wasm]`). `SandboxedAgent` transparent wrapper. `create_sandbox_runner()` factory. `MountAllowlist` with path traversal prevention.
- **Scheduler** (`scheduler/`) — `TaskScheduler` with cron/interval/once scheduling, SQLite persistence, 5 MCP tools, event bus. CLI: `jarvis scheduler create|list|pause|resume|cancel|logs|start`.
- **Agent Hardening** (`agents/loop_guard.py`) — `LoopGuard`: SHA-256 hash tracking (identical calls), ping-pong detection (A-B-A-B patterns), poll-tool budget, context overflow recovery. `BaseAgent._check_continuation()` auto-resumes on `finish_reason=length`.
- **Workflow Engine** (`workflow/`) — DAG-based `WorkflowGraph` (cycle detection, topological sort, parallel stages via `ThreadPoolExecutor`). `WorkflowBuilder` fluent API. `WorkflowEngine` executes against `JarvisSystem`. TOML loader. Node types: agent, tool, condition, parallel, loop, transform.
- **Skills** (`skills/`) — `SkillManifest`/`SkillExecutor` (sequential tool steps with template rendering). Ed25519 signature verification. `SkillTool` adapter wraps skills as invocable tools. TOML loader.
- **Knowledge Graph** (`tools/storage/knowledge_graph.py`) — `KnowledgeGraphMemory(MemoryBackend)`: SQLite entity-relation store. `add_entity()`, `add_relation()`, `neighbors()`, `query_pattern()`. Registered as `"knowledge_graph"`.
- **Sessions** (`sessions/`) — `SessionStore` (SQLite): cross-channel persistent sessions. `SessionIdentity` canonical user across channels. `consolidate()` summarizes old messages, `decay()` removes expired.
- **A2A Protocol** (`a2a/`) — Google Agent-to-Agent spec (JSON-RPC 2.0). `A2AServer` (tasks/send, tasks/get, tasks/cancel, `/.well-known/agent.json`). `A2AClient`. `A2AAgentTool` adapter.
- **TUI Dashboard** (`cli/dashboard.py`) — `textual`-based terminal dashboard (optional `[dashboard]`). Panels: system status, event stream, telemetry, agent activity, sessions.
- **Desktop App** (`desktop/`) — Tauri 2.0 native desktop application. 5 dashboard panels: EnergyDashboard (real-time power monitoring with recharts), TraceDebugger (timeline inspection with step-type color coding), LearningCurve (policy visualization, GRPO/bandit stats), MemoryBrowser (search + stats), AdminPanel (health, agents, server control). Tauri commands proxy to OpenJarvis REST API. Plugins: notification, shell, global-shortcut, autostart, updater, single-instance. CI: `.github/workflows/desktop.yml` (Linux/macOS/Windows).
- **Vault** (`cli/vault_cmd.py`) — Fernet-encrypted credential store at `~/.openjarvis/vault.enc` with auto-generated key (`0o600` permissions).
- **MCP Quick-Add** (`cli/add_cmd.py`) — `jarvis add <server>` with 8 templates (github, filesystem, slack, postgres, brave-search, memory, puppeteer, google-maps). Saves JSON config to `~/.openjarvis/mcp/`.

### Core Module (`src/openjarvis/core/`)

- `registry.py` — `RegistryBase[T]` generic base. Subclasses: `ModelRegistry`, `EngineRegistry`, `MemoryRegistry`, `AgentRegistry`, `ToolRegistry`, `RouterPolicyRegistry`, `BenchmarkRegistry`, `ChannelRegistry`, `LearningRegistry`, `SkillRegistry`.
- `types.py` — `Message`, `Conversation`, `ModelSpec`, `ToolResult`, `TelemetryRecord`, `StepType`, `TraceStep`, `Trace`, `RoutingContext`.
- `config.py` — `JarvisConfig` dataclass hierarchy with TOML loader. Config classes for each pillar/subsystem. TOML sections: `[engine]` (+ nested `[engine.ollama]`, `[engine.vllm]`, `[engine.sglang]`, `[engine.llamacpp]`, `[engine.mlx]`, `[engine.lmstudio]`), `[intelligence]`, `[agent]`, `[tools.storage]`, `[tools.mcp]`, `[tools.browser]`, `[learning]` (+ nested routing/intelligence/agent/metrics), `[server]`, `[telemetry]`, `[traces]`, `[channel]`, `[security]` (+ `[security.capabilities]`, `ssrf_protection`, `rate_limit_*`), `[sandbox]`, `[scheduler]`, `[workflow]`, `[sessions]`, `[a2a]`. Backward-compat: `engine.ollama_host` → `engine.ollama.host`, `agent.default_tools` → `agent.tools`, TOML migration for cross-section moves.
- `events.py` — Pub/sub event bus (synchronous dispatch). ~30 EventType values covering inference, tools, memory, agents, telemetry, traces, channels, security, scheduler, workflow, skills, sessions, A2A.

### Docker & Deployment

- `deploy/docker/Dockerfile` — Multi-stage: Python 3.12-slim, `.[server]`, entrypoint `jarvis serve`
- `deploy/docker/Dockerfile.gpu` — NVIDIA CUDA 12.4 variant
- `deploy/docker/Dockerfile.gpu.rocm` — AMD ROCm 6.2 variant
- `deploy/docker/docker-compose.yml` — `jarvis` (8000) + `ollama` (11434). ROCm override: `docker-compose.gpu.rocm.yml`
- `deploy/systemd/openjarvis.service`, `deploy/launchd/com.openjarvis.plist`

### Query Flow

User query → Security scanning (input) → Intelligence resolves model → Agentic Logic (tools/memory) → Memory retrieval → Context injection → Engine generates → Security scanning (output) → Trace recorded → Telemetry recorded → Learning policies update.

### API Surface

OpenAI-compatible server via `jarvis serve`:
- **Core**: `POST /v1/chat/completions`, `GET /v1/models`, `GET /health`
- **Channels**: `GET /v1/channels`, `POST /v1/channels/send`, `GET /v1/channels/status`
- **Agents**: `GET /v1/agents`, `POST /v1/agents`, `DELETE /v1/agents/{id}`, `POST /v1/agents/{id}/message`
- **Memory**: `POST /v1/memory/store`, `POST /v1/memory/search`, `GET /v1/memory/stats`
- **Traces**: `GET /v1/traces`, `GET /v1/traces/{id}`
- **Telemetry**: `GET /v1/telemetry/stats`, `GET /v1/telemetry/energy`
- **Learning**: `GET /v1/learning/stats`, `GET /v1/learning/policy`
- **Skills**: `GET /v1/skills`, `POST /v1/skills`, `DELETE /v1/skills/{name}`
- **Speech**: `POST /v1/speech/transcribe`, `GET /v1/speech/health`
- **Sessions**: `GET /v1/sessions`, `GET /v1/sessions/{id}`
- **Budget**: `GET /v1/budget`, `PUT /v1/budget/limits`
- **Metrics**: `GET /metrics` (Prometheus-compatible)
- **WebSocket**: `WS /v1/chat/stream` (JSON chunked streaming)
- SSE streaming on `/v1/chat/completions` with `stream=true`

## Key Design Patterns

- **Registry pattern:** All extensible components use `@XRegistry.register("name")` decorator for registration and runtime discovery.
- **ABC interfaces:** Each pillar defines an ABC. Implement the ABC + register via decorator to add a new backend.
- **Offline-first:** Cloud APIs are optional. All core functionality works without network.
- **Hardware-aware:** Auto-detect GPU vendor/model/VRAM via `nvidia-smi`, `rocm-smi`, `system_profiler`, `/proc/cpuinfo`. Recommend engine accordingly.
- **Telemetry opt-in:** `InstrumentedEngine` wraps inference transparently. Agents unaware of telemetry.
- **Backward-compat shims:** `intelligence/` re-exports from `learning/`, `agents/react.py` re-exports as `ReActAgent`, registry alias `"react"` → `NativeReActAgent`. Old config keys continue to work.
- **`ensure_registered()` pattern:** Benchmark and learning modules use lazy registration to survive registry clearing in tests.

## Development Phases

| Version | Phase | Delivers |
|---------|-------|----------|
| v0.1 | 0 | Scaffolding, registries, core types, config, CLI skeleton |
| v0.2 | 1 | Intelligence + Inference — `jarvis ask` end-to-end |
| v0.3 | 2 | Memory backends, document indexing, context injection |
| v0.4 | 3 | Agents, tool system, OpenAI-compatible API server |
| v0.5 | 4 | Learning, telemetry aggregation, `--router` CLI |
| v1.0 | 5 | SDK, OpenClaw infra, benchmarks, Docker |
| v1.1 | 6 | Trace system, trace-driven learning, pluggable agents |
| v1.2 | 7 | 5-pillar restructuring, composition layer, MCP, structured learning |
| v1.3 | 8 | Intelligence = "The Model", routing → Learning, engine selection |
| v1.4 | 9 | Pillar-aligned config, nested configs, TOML migration |
| v1.5 | 10 | Agent restructuring, BaseAgent/ToolUsingAgent, `accepts_tools`, OpenHands SDK |
| v1.6 | 11 | NanoClaw subsumption: ClaudeCodeAgent, WhatsApp Baileys, Docker sandbox, TaskScheduler |
| v1.7 | 12 | EnergyMonitor ABC (NVIDIA/AMD/Apple/RAPL), EnergyBatch, SteadyStateDetector |
| v1.8 | 13 | `jarvis doctor`/`init`, MLX engine, AMD multi-GPU, PWA, ROCm Docker |
| v1.9 | 14 | Agent hardening: LoopGuard, RBAC CapabilityPolicy, taint tracking, Merkle audit, Ed25519 |
| v2.0 | 15 | WorkflowEngine (DAG), SkillSystem, KnowledgeGraphMemory, SessionStore |
| v2.1 | 16 | A2A protocol, MCP templates, WasmRunner, TUI dashboard |
| v2.2 | 17 | Production tool parity: FileWrite, ApplyPatch, ShellExec, Git, HTTP, DB, Browser, Agent, Media, PDF tools. SSRF protection, injection scanner, rate limiter, subprocess sandbox, security middleware |
| v2.3 | 18 | CLI expansion (20 commands): daemon, chat REPL, agent, workflow, skill, vault, add. API expansion (40+ endpoints): agents, memory, traces, telemetry, learning, skills, sessions, budget, metrics, WebSocket streaming |
| v2.4 | 19 | Learning productionization: GRPO (softmax/advantage), BanditRouter (Thompson/UCB1), SkillDiscovery (trace mining), ICL updates (versioning/rollback/quality gates) |
| v2.5 | 20 | Tauri 2.0 desktop app: energy dashboard, trace debugger, learning curve visualization, memory browser, admin panel. CI for Linux/macOS/Windows |
| v2.6 | 21 | 10 new channels: LINE, Viber, Messenger, Reddit, Mastodon, XMPP, Rocket.Chat, Zulip, Twitch, Nostr |
| v2.7 | 22 | Operators: persistent, scheduled autonomous agents with recipe + schedule + channel output |
| v2.8 | 23 | Differentiated functionalities: trace-driven learning pipeline (TrainingDataMiner, LoRATrainer, AgentConfigEvolver, LearningOrchestrator), 15 real IPW benchmarks, composable recipes, 15 agent templates, 20 bundled skills, 3 operator recipes |
| v2.9 | 24 | Speech subsystem: SpeechBackend ABC, SpeechRegistry, 3 backends (FasterWhisper local, OpenAI cloud, Deepgram cloud), auto-discovery, API endpoints, frontend MicButton + useSpeech hook, Tauri commands, SystemBuilder wiring |
| v3.0 | 25 | Operator benchmarks: LogHub, AMA-Bench, LifelongAgentBench, WebChoreArena, WorkArena++. MonitorOperativeAgent with 4 configurable strategies (memory extraction, observation compression, retrieval, task decomposition). EvalRunner episode mode. EnvironmentProvider ABC. browser_axtree tool |
