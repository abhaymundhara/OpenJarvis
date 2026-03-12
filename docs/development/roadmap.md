# Roadmap

OpenJarvis development follows a phased approach, with each version adding
a major primitive or cross-cutting capability to the framework.

---

## Development Phases

| Version | Phase | Status | Delivers |
|---|---|---|---|
| **v0.1** | Phase 0 -- Scaffolding | :material-check-circle:{ .green } Complete | Project scaffolding, registry system (`RegistryBase[T]`), core types (`Message`, `ModelSpec`, `Conversation`, `ToolResult`), configuration loader with hardware detection, Click CLI skeleton |
| **v0.2** | Phase 1 -- Intelligence + Inference | :material-check-circle:{ .green } Complete | Intelligence primitive (model catalog, heuristic router), inference engines (Ollama, vLLM, llama.cpp), engine discovery and health probing, `jarvis ask` command working end-to-end |
| **v0.3** | Phase 2 -- Memory | :material-check-circle:{ .green } Complete | Memory backends (SQLite/FTS5, FAISS, ColBERTv2, BM25, Hybrid/RRF), document chunking and ingestion pipeline, context injection with source attribution, `jarvis memory` commands |
| **v0.4** | Phase 3 -- Agents + Tools + Server | :material-check-circle:{ .green } Complete | Agent system (SimpleAgent, OrchestratorAgent), tool system (Calculator, Think, Retrieval, LLM, FileRead), ToolExecutor dispatch engine, OpenAI-compatible API server (`jarvis serve`) |
| **v0.5** | Phase 4 -- Learning + Telemetry | :material-check-circle:{ .green } Complete | Learning system (HeuristicRouter policy, TraceDrivenPolicy, GRPO stub), reward functions, telemetry aggregation (per-model/engine stats, export), `--router` CLI flag, `jarvis telemetry` commands |
| **v1.0** | Phase 5 -- SDK + Production | :material-check-circle:{ .green } Complete | Python SDK (`Jarvis` class, `MemoryHandle`), multi-platform channel system (Telegram, Discord, Slack, WhatsApp, etc.), benchmarking framework (latency, throughput), Docker deployment (CPU + GPU), MkDocs documentation site |
| **v1.1** | Phase 6 -- Traces + Learning | :material-check-circle:{ .green } Complete | Trace system (`TraceStore`, `TraceCollector`, `TraceAnalyzer`), trace-driven learning, MCP integration layer |
| **v1.5** | Phase 10 -- Agent Restructuring | :material-check-circle:{ .green } Complete | BaseAgent helpers, ToolUsingAgent intermediate base, NativeReActAgent, NativeOpenHandsAgent, RLMAgent, OpenHandsAgent (SDK), `accepts_tools` introspection, backward-compat shims, CustomAgent removed |

---

## Current Status

OpenJarvis v1.5 (Phase 10) is complete. The framework provides:

- **Four core abstractions** -- Intelligence, Engine, Agentic Logic, Memory -- each with an ABC interface and registry-based discovery
- **Five inference engines** -- Ollama, vLLM, llama.cpp, SGLang, Cloud (OpenAI/Anthropic/Google)
- **Five memory backends** -- SQLite/FTS5, FAISS, ColBERTv2, BM25, Hybrid (RRF fusion)
- **Seven agent types** -- Simple, Orchestrator, NativeReAct, NativeOpenHands, RLM, Operative, MonitorOperative
- **Seven built-in tools** -- Calculator, Think, Retrieval, LLM, FileRead, WebSearch, CodeInterpreter
- **Python SDK** -- `Jarvis` class for programmatic use
- **OpenAI-compatible API server** -- `POST /v1/chat/completions`, `GET /v1/models`
- **Benchmarking framework** -- Latency and throughput measurements
- **Telemetry and traces** -- SQLite-backed recording and aggregation
- **Docker deployment** -- CPU and GPU images with docker-compose

Phase 10 (Agent Restructuring) is complete. The agent hierarchy has been
refactored with `BaseAgent` helpers, `ToolUsingAgent` intermediate base, and
four new agent types (NativeReActAgent, NativeOpenHandsAgent, RLMAgent,
OpenHandsAgent SDK).

---

## Phase 10 Details

Phase 10 refactored the agent hierarchy for composability and extensibility:

### BaseAgent Helpers

- **`_emit_turn_start` / `_emit_turn_end`** -- Event bus integration without boilerplate
- **`_build_messages`** -- System prompt + context + input assembly
- **`_generate`** -- Engine call with stored defaults
- **`_max_turns_result`** -- Standard max-turns-exceeded result
- **`_strip_think_tags`** -- Remove `<think>` blocks from model output

### ToolUsingAgent Intermediate Base

- Sets `accepts_tools = True` for CLI/SDK introspection
- Initializes `ToolExecutor` from provided tools
- Configurable `max_turns` loop limit

### New Agent Types

- **NativeReActAgent** (`native_react`, alias `react`) -- Thought-Action-Observation loop
- **NativeOpenHandsAgent** (`native_openhands`) -- CodeAct-style code execution with URL pre-fetching
- **RLMAgent** (`rlm`) -- Recursive LM with persistent REPL and sub-LM calls
- **OpenHandsAgent** (`openhands`) -- Thin wrapper for real `openhands-sdk`

---

## Future Directions

Beyond Phase 10, areas of ongoing exploration include:

- **GRPO training** -- Reinforcement learning from trace data to train the
  routing policy, moving beyond heuristics and simple statistics
- **Streaming telemetry** -- Real-time performance dashboards and alerting
- **Multi-model orchestration** -- Coordinating multiple models within a
  single query pipeline (e.g., small model for classification, large model
  for generation)
- **Federated memory** -- Memory backends that synchronize across devices
- **Plugin ecosystem** -- Community-contributed engines, tools, and agents
  distributed as Python packages
- **Energy-aware routing** -- Using power consumption data from telemetry to
  optimize for energy efficiency alongside latency and quality
