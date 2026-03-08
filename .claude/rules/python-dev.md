# Python Development

## Architecture: The Five Pillars

All pillars are wired together by `JarvisSystem` (`src/openjarvis/system.py`) which is constructed from `configs/openjarvis/config.toml` (or `~/.openjarvis/config.toml`).

### 1. Intelligence (`src/openjarvis/intelligence/`)
Model selection, provider routing. Config section: `[intelligence]`.

### 2. Agent (`src/openjarvis/agents/`)
Multi-turn reasoning and tool use. Agents register via `@AgentRegistry.register()` decorator. Key agents: `simple`, `native_react`, `native_openhands`, `orchestrator`, `monitor_operative`, `claude_code`, `rlm`. Config section: `[agent]`.

### 3. Tools (`src/openjarvis/tools/`)
Built-in tools (code_interpreter, web_search, file_read, shell_exec, calculator, think, browser, etc.) plus MCP adapter. Tool storage backends in `tools/storage/`. Config section: `[tools]`.

### 4. Engine (`src/openjarvis/engine/`)
Inference runtime abstraction. All engines implement `InferenceEngine` (defined in `engine/_stubs.py`) and use OpenAI-compatible chat completions. Supported: vLLM, Ollama, llama.cpp, SGLang, MLX, cloud (OpenAI/Anthropic/Google), LiteLLM, Apple FM, Exo, Nexa. Discovery in `engine/_discovery.py`. Config section: `[engine]`.

### 5. Learning (`src/openjarvis/learning/`)
Improvement methodologies: router policies (heuristic, bandit, trace-based), SFT/GRPO training, ICL updater, agent evolution, skill discovery. Orchestrated by `LearningOrchestrator`. Config section: `[learning]`.

### Supporting Systems
- **Core** (`core/`): `RegistryBase` pattern (decorator-based registration), types (`Message`, `Conversation`, `ToolCall`), config loader with hardware detection, `EventBus`.
- **Channels** (`channels/`): Chat platform integrations (Telegram, Discord, Slack, WhatsApp, Signal, IRC, Matrix, etc.).
- **Telemetry** (`telemetry/`): GPU monitoring, energy measurement (NVIDIA/AMD/Apple/RAPL), latency instrumentation, vLLM metrics.
- **Traces** (`traces/`): Execution trace recording for analysis.
- **MCP** (`mcp/`): Model Context Protocol server.
- **Security** (`security/`): PII scanning, capability policies.
- **Server** (`server/`): FastAPI REST API.
- **SDK** (`sdk.py`): High-level `Jarvis` and `JarvisSystem` classes, `MemoryHandle` for memory operations.

## Key Patterns

- **Registry pattern**: Components (agents, engines, memory backends, tools, channels, etc.) self-register via `@XRegistry.register("key")` decorators. Tests auto-clear all registries via `conftest.py` fixture.
- **Optional dependencies**: Heavy deps are extras in `pyproject.toml` (e.g., `inference-cloud`, `memory-faiss`, `channel-telegram`). Import failures are caught with try/except so the core stays lightweight.
- **OpenAI-compatible**: All engines expose an OpenAI-format chat completions interface. `messages_to_dicts()` in `engine/_base.py` handles conversion.
- **Config-driven**: TOML configs control everything. `load_config()` detects hardware, fills defaults, then overlays user overrides.
