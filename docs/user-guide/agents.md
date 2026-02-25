# Agents

Agents are the agentic logic layer of OpenJarvis. They determine how a query is processed -- whether it goes directly to a model, through a tool-calling loop, via ReAct reasoning, CodeAct code execution, recursive decomposition, or an external agent runtime. All agents implement the `BaseAgent` ABC and are registered via the `AgentRegistry`.

## Overview

| Agent               | Registry Key      | `accepts_tools` | Multi-turn | Description                                  |
|---------------------|-------------------|-----------------|------------|----------------------------------------------|
| `SimpleAgent`       | `simple`          | No              | No         | Single-turn query-to-response                |
| `OrchestratorAgent` | `orchestrator`    | Yes             | Yes        | Multi-turn tool-calling loop (function_calling + structured) |
| `NativeReActAgent`  | `native_react`    | Yes             | Yes        | Thought-Action-Observation loop              |
| `NativeOpenHandsAgent` | `native_openhands` | Yes          | Yes        | CodeAct-style code execution + tool calls    |
| `RLMAgent`          | `rlm`             | Yes             | Yes        | Recursive LM with persistent REPL            |
| `OpenHandsAgent`    | `openhands`       | No              | Yes        | Wraps real openhands-sdk                     |
| `OpenClawAgent`     | `openclaw`        | Yes             | Yes        | External agent via HTTP or subprocess         |
| `ClaudeCodeAgent`   | `claude_code`     | No              | Yes        | Claude Agent SDK via Node.js subprocess       |

---

## BaseAgent ABC

All agents extend the abstract `BaseAgent` class.

```python
from abc import ABC, abstractmethod
from openjarvis.agents._stubs import AgentContext, AgentResult

class BaseAgent(ABC):
    agent_id: str
    accepts_tools: bool = False

    def __init__(
        self,
        engine: InferenceEngine,
        model: str,
        *,
        bus: Optional[EventBus] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> None: ...

    @abstractmethod
    def run(
        self,
        input: str,
        context: AgentContext | None = None,
        **kwargs,
    ) -> AgentResult:
        """Execute the agent on the given input."""
```

The `accepts_tools` class attribute controls whether an agent can receive tools via `--tools` on the CLI or `tools=` in the SDK. Agents with `accepts_tools = False` ignore tool arguments.

`BaseAgent` also provides concrete helper methods (`_emit_turn_start`, `_emit_turn_end`, `_build_messages`, `_generate`, `_max_turns_result`, `_strip_think_tags`) that subclasses use to avoid duplicating common logic. See the [architecture docs](../architecture/agents.md#baseagent-abc) for details.

**ToolUsingAgent** is an intermediate base class (extends `BaseAgent`) that sets `accepts_tools = True` and adds a `ToolExecutor` and `max_turns` loop limit. All tool-using agents extend this class.

### AgentContext

The runtime context handed to an agent on each invocation.

| Field            | Type               | Description                                    |
|------------------|--------------------|------------------------------------------------|
| `conversation`   | `Conversation`     | Message history (pre-filled with context if memory injection is active) |
| `tools`          | `list[str]`        | Tool names available to the agent              |
| `memory_results` | `list[Any]`        | Pre-fetched memory retrieval results           |
| `metadata`       | `dict[str, Any]`   | Arbitrary metadata for the run                 |

### AgentResult

The result returned after an agent completes a run.

| Field          | Type               | Description                                    |
|----------------|--------------------|------------------------------------------------|
| `content`      | `str`              | The final response text                        |
| `tool_results` | `list[ToolResult]` | Results from tool executions during the run    |
| `turns`        | `int`              | Number of turns (inference calls) taken        |
| `metadata`     | `dict[str, Any]`   | Arbitrary metadata about the run               |

---

## SimpleAgent

The `SimpleAgent` is a single-turn agent that sends the query directly to the inference engine and returns the response. It does not support tool calling.

**How it works:**

1. Builds a message list from the conversation context (if provided) plus the user query.
2. Calls the inference engine via `_generate()`.
3. Returns the response as an `AgentResult` with `turns=1`.

**Constructor parameters:**

| Parameter     | Type              | Default | Description                        |
|---------------|-------------------|---------|------------------------------------|
| `engine`      | `InferenceEngine` | --      | The inference engine to use        |
| `model`       | `str`             | --      | Model identifier                   |
| `bus`         | `EventBus`        | `None`  | Event bus for telemetry            |
| `temperature` | `float`           | `0.7`   | Sampling temperature               |
| `max_tokens`  | `int`             | `1024`  | Maximum tokens to generate         |

**When to use:** For straightforward question-answering without tool calling or multi-turn reasoning.

---

## OrchestratorAgent

The `OrchestratorAgent` is a multi-turn agent that implements a tool-calling loop. It is the primary agent for queries that require computation, knowledge retrieval, or structured reasoning. Extends `ToolUsingAgent`.

**How it works:**

1. Builds the initial message list from context and the user query.
2. Sends messages with tool definitions (OpenAI function-calling format) to the engine.
3. If the engine responds with `tool_calls`, the `ToolExecutor` dispatches each call.
4. Tool results are appended as `TOOL` messages and the loop continues.
5. If no `tool_calls` are returned, the response is treated as the final answer.
6. The loop stops after `max_turns` iterations (default: 10), returning whatever content is available along with a `max_turns_exceeded` metadata flag.

**Constructor parameters:**

| Parameter       | Type              | Default | Description                          |
|-----------------|-------------------|---------|--------------------------------------|
| `engine`        | `InferenceEngine` | --      | The inference engine to use          |
| `model`         | `str`             | --      | Model identifier                     |
| `tools`         | `list[BaseTool]`  | `[]`    | Tool instances to make available     |
| `bus`           | `EventBus`        | `None`  | Event bus for telemetry              |
| `max_turns`     | `int`             | `10`    | Maximum number of tool-calling turns |
| `temperature`   | `float`           | `0.7`   | Sampling temperature                 |
| `max_tokens`    | `int`             | `1024`  | Maximum tokens to generate           |
| `mode`          | `str`             | `"function_calling"` | Tool-calling mode (`function_calling` or `structured`) |
| `system_prompt` | `str`             | `None`  | Custom system prompt                 |

**When to use:** For queries that need calculation, memory search, sub-model calls, file reading, or multi-step reasoning.

!!! info "Tool-Calling Loop"
    The orchestrator follows the OpenAI function-calling convention. The engine must support returning `tool_calls` in its response for the loop to engage. If tools are provided but the engine does not return any tool calls, the agent behaves like a single-turn agent.

---

## NativeReActAgent

The `NativeReActAgent` implements a **Thought-Action-Observation** loop following the ReAct pattern. It prompts the LLM to produce structured output (`Thought:`, `Action:`, `Action Input:`, `Final Answer:`) and parses the response to drive tool execution. Extends `ToolUsingAgent`.

**How it works:**

1. Builds a system prompt with enriched tool descriptions (names, parameter schemas, categories) via `build_tool_descriptions()`. Parsing is case-insensitive.
2. Generates a response and parses the ReAct-structured output.
3. If a `Final Answer:` is found, returns it.
4. If an `Action:` is found, executes the tool and feeds the result back as an `Observation:`.
5. Loops until a final answer is produced or `max_turns` is exceeded.

**Constructor parameters:**

| Parameter     | Type              | Default | Description                        |
|---------------|-------------------|---------|------------------------------------|
| `engine`      | `InferenceEngine` | --      | The inference engine to use        |
| `model`       | `str`             | --      | Model identifier                   |
| `tools`       | `list[BaseTool]`  | `[]`    | Tool instances to make available   |
| `bus`         | `EventBus`        | `None`  | Event bus for telemetry            |
| `max_turns`   | `int`             | `10`    | Maximum number of reasoning turns  |
| `temperature` | `float`           | `0.7`   | Sampling temperature               |
| `max_tokens`  | `int`             | `1024`  | Maximum tokens to generate         |

**When to use:** For queries that benefit from explicit step-by-step reasoning with tool use, where you want visibility into the agent's thought process.

!!! note "Backward compatibility"
    The registry alias `"react"` maps to `NativeReActAgent`. The old import `from openjarvis.agents.react import ReActAgent` also still works.

---

## NativeOpenHandsAgent

The `NativeOpenHandsAgent` is a CodeAct-style agent that generates and executes Python code alongside structured tool calls. It can also pre-fetch URL content from user input to provide direct context to the LLM. Extends `ToolUsingAgent`.

**How it works:**

1. Builds a detailed system prompt with enriched tool descriptions (via shared `build_tool_descriptions()` builder) and code execution instructions.
2. Pre-fetches any URLs in the user input, inlining the content directly.
3. For each turn, generates a response and attempts to extract code blocks or tool calls.
4. Code is executed via `code_interpreter`; tool calls are dispatched via `ToolExecutor`.
5. If neither is found, returns the content as the final answer.

**Constructor parameters:**

| Parameter     | Type              | Default | Description                        |
|---------------|-------------------|---------|------------------------------------|
| `engine`      | `InferenceEngine` | --      | The inference engine to use        |
| `model`       | `str`             | --      | Model identifier                   |
| `tools`       | `list[BaseTool]`  | `[]`    | Tool instances to make available   |
| `bus`         | `EventBus`        | `None`  | Event bus for telemetry            |
| `max_turns`   | `int`             | `3`     | Maximum number of turns            |
| `temperature` | `float`           | `0.7`   | Sampling temperature               |
| `max_tokens`  | `int`             | `2048`  | Maximum tokens to generate         |

**When to use:** For queries involving URL content, code execution, or tasks where the LLM can write and run Python to solve the problem.

---

## RLMAgent

The `RLMAgent` implements recursive decomposition via a persistent REPL, based on the RLM paper. Context is stored as a Python variable rather than injected into the prompt, enabling processing of arbitrarily long inputs through recursive sub-LM calls. Extends `ToolUsingAgent`.

**How it works:**

1. Creates a persistent REPL with `llm_query()` and `llm_batch()` callbacks.
2. Injects context from `AgentContext` into the REPL as a variable.
3. Generates code and executes it in the REPL.
4. If `FINAL(value)` is called, returns the value as the final answer.
5. If no code block is found, treats the content as a direct text answer.

**Constructor parameters:**

| Parameter          | Type              | Default            | Description                        |
|--------------------|-------------------|--------------------|-------------------------------------|
| `engine`           | `InferenceEngine` | --                 | The inference engine to use        |
| `model`            | `str`             | --                 | Model identifier                   |
| `tools`            | `list[BaseTool]`  | `[]`               | Tool instances (optional)          |
| `bus`              | `EventBus`        | `None`             | Event bus for telemetry            |
| `max_turns`        | `int`             | `10`               | Maximum number of code-execute turns |
| `temperature`      | `float`           | `0.7`              | Sampling temperature               |
| `max_tokens`       | `int`             | `2048`             | Maximum tokens to generate         |
| `sub_model`        | `str`             | same as `model`    | Model for sub-LM calls            |
| `sub_temperature`  | `float`           | `0.3`              | Temperature for sub-LM calls       |
| `sub_max_tokens`   | `int`             | `1024`             | Max tokens for sub-LM calls        |
| `max_output_chars` | `int`             | `10000`            | Max REPL output characters         |
| `system_prompt`    | `str`             | `RLM_SYSTEM_PROMPT` | Override the system prompt         |

**When to use:** For long-context tasks that benefit from recursive decomposition, such as summarizing large documents, processing structured data, or tasks that require programmatic manipulation of context.

---

## OpenHandsAgent (SDK)

The `OpenHandsAgent` wraps the real `openhands-sdk` package for AI-driven software development. Extends `BaseAgent` directly (tool management is handled by the SDK internally).

**How it works:**

1. Imports `openhands.sdk` at runtime.
2. Creates an LLM, Agent, and Conversation from the SDK.
3. Sends the input and runs the conversation.
4. Returns the final message content.

**Constructor parameters:**

| Parameter     | Type              | Default       | Description                        |
|---------------|-------------------|---------------|------------------------------------|
| `engine`      | `InferenceEngine` | --            | The inference engine (fallback)    |
| `model`       | `str`             | --            | Model identifier                   |
| `bus`         | `EventBus`        | `None`        | Event bus for telemetry            |
| `temperature` | `float`           | `0.7`         | Sampling temperature               |
| `max_tokens`  | `int`             | `1024`        | Maximum tokens to generate         |
| `workspace`   | `str`             | `os.getcwd()` | Working directory for the agent    |
| `api_key`     | `str`             | `$LLM_API_KEY`| API key for the LLM provider      |

**When to use:** For software development tasks (debugging, code editing, test fixing) where the OpenHands SDK provides a full development agent runtime.

!!! warning "Optional dependency"
    Requires `openhands-sdk` (`pip install openjarvis[openhands]`) and Python 3.12+.

---

## OpenClawAgent

The `OpenClawAgent` wraps the OpenClaw Pi agent runtime, communicating via either HTTP or subprocess transport. It supports tool calling through the OpenClaw protocol.

**How it works:**

1. Checks transport health.
2. Sends a `QUERY` protocol message through the transport.
3. If the response is a `TOOL_CALL`, dispatches the tool locally via `ToolExecutor`.
4. Sends the tool result back as a `TOOL_RESULT` message.
5. Continues the tool-call loop until the response is a final answer or error (up to 10 turns).

**Constructor parameters:**

| Parameter    | Type                 | Default   | Description                               |
|--------------|----------------------|-----------|-------------------------------------------|
| `engine`     | `Any`                | `None`    | Inference engine (fallback/provider)       |
| `model`      | `str`                | `""`      | Model identifier                          |
| `transport`  | `OpenClawTransport`  | `None`    | Pre-configured transport (overrides mode)  |
| `mode`       | `str`                | `"http"`  | Transport mode: `"http"` or `"subprocess"` |
| `bus`        | `EventBus`           | `None`    | Event bus for telemetry                   |

**Transport modes:**

- **HTTP** (`HttpTransport`): Sends HTTP POST requests to an OpenClaw server.
- **Subprocess** (`SubprocessTransport`): Spawns a Node.js process and communicates via stdin/stdout using JSON-line protocol.

!!! warning "Node.js Requirement"
    The subprocess transport mode requires Node.js 22+ to be installed on the system.

---

## Using Agents

### Via CLI

```bash
# Simple agent
jarvis ask --agent simple "What is the capital of France?"

# Orchestrator with tools
jarvis ask --agent orchestrator --tools calculator,think "What is sqrt(256)?"

# NativeReActAgent
jarvis ask --agent native_react --tools calculator "What is 2+2?"

# ReAct alias (same as native_react)
jarvis ask --agent react --tools calculator,think "Solve step by step: 15% of 340"

# NativeOpenHandsAgent
jarvis ask --agent native_openhands --tools calculator,web_search "Summarize example.com"

# RLMAgent
jarvis ask --agent rlm "Summarize this long document"

# OpenHands SDK agent
jarvis ask --agent openhands "Fix the bug in test_utils.py"

# OpenClaw agent
jarvis ask --agent openclaw "Tell me a story"
```

### Via Python SDK

```python
from openjarvis import Jarvis

j = Jarvis()

# Simple agent
response = j.ask("Hello", agent="simple")

# Orchestrator with tools
response = j.ask(
    "Calculate 15% of 340",
    agent="orchestrator",
    tools=["calculator"],
)

# NativeReActAgent with tools
response = j.ask(
    "What is sqrt(256)?",
    agent="native_react",
    tools=["calculator", "think"],
)

# Full result with tool details
result = j.ask_full(
    "What is the square root of 144?",
    agent="orchestrator",
    tools=["calculator", "think"],
)
print(result["content"])
print(result["turns"])
print(result["tool_results"])

j.close()
```

---

---

## ClaudeCodeAgent

The `ClaudeCodeAgent` wraps the `@anthropic-ai/claude-code` SDK via a bundled Node.js subprocess bridge. Unlike the other agents, inference is handled entirely by the Claude Agent SDK -- the `engine` parameter is accepted only for `BaseAgent` interface conformance and is not used.

!!! warning "Requirements"
    Requires Node.js 22+ on `PATH` and an `ANTHROPIC_API_KEY` environment variable (or pass `api_key=` directly). The bundled runner is auto-installed to `~/.openjarvis/claude_code_runner/` on first use via `npm install`.

**How it works:**

1. On first call, copies the bundled `claude_code_runner/` to `~/.openjarvis/claude_code_runner/` and runs `npm install --production` if `node_modules` is missing.
2. Builds a JSON request payload (prompt, API key, workspace, allowed tools, system prompt, session ID) and sends it to `stdin` of a `node dist/index.js` subprocess.
3. The Node.js runner calls the Claude Agent SDK and writes sentinel-delimited JSON to `stdout`.
4. The Python side parses the output between `---OPENJARVIS_OUTPUT_START---` and `---OPENJARVIS_OUTPUT_END---` markers, extracting content, tool results, and metadata.
5. Returns an `AgentResult` with `turns=1`.

**Constructor parameters:**

| Parameter        | Type              | Default             | Description                                      |
|------------------|-------------------|---------------------|--------------------------------------------------|
| `engine`         | `InferenceEngine` | --                  | Accepted for interface conformance; not used     |
| `model`          | `str`             | --                  | Accepted for interface conformance; not used     |
| `bus`            | `EventBus`        | `None`              | Event bus for telemetry                          |
| `temperature`    | `float`           | `0.7`               | Accepted for interface conformance; not used     |
| `max_tokens`     | `int`             | `1024`              | Accepted for interface conformance; not used     |
| `api_key`        | `str`             | `$ANTHROPIC_API_KEY`| Anthropic API key                                |
| `workspace`      | `str`             | `os.getcwd()`       | Working directory for the Claude agent           |
| `session_id`     | `str`             | `""`                | Optional session ID for conversation continuity  |
| `allowed_tools`  | `list[str]`       | `None` (all)        | Claude Code tool names to allow                  |
| `system_prompt`  | `str`             | `""`                | Additional system prompt for the agent           |
| `timeout`        | `int`             | `300`               | Subprocess timeout in seconds                    |

**When to use:** For software engineering tasks where the Claude Agent SDK's built-in tools (code editing, bash execution, file operations) provide capabilities beyond what OpenJarvis tool-calling agents support.

```python
from openjarvis.agents.claude_code import ClaudeCodeAgent

agent = ClaudeCodeAgent(
    engine=None,          # not used
    model="",             # not used
    workspace="/path/to/project",
    allowed_tools=["Read", "Write", "Bash"],
    timeout=120,
)
result = agent.run("Add type hints to all functions in utils.py")
print(result.content)
```

```bash
# Via CLI
jarvis ask --agent claude_code "Refactor the tests to use pytest fixtures"
```

!!! info "accepts_tools = False"
    `ClaudeCodeAgent` does not accept OpenJarvis tools via `--tools`. Tool access for the Claude agent is configured separately via the `allowed_tools` constructor parameter, which passes tool names understood by the Claude Agent SDK itself.

---

## SandboxedAgent

`SandboxedAgent` is a transparent wrapper that runs **any** `BaseAgent` inside a Docker (or Podman) container. It follows the same wrapper pattern as `GuardrailsEngine` -- the inner agent's configuration is serialized and sent to the container's stdin, and the result is read back from stdout.

See also the [`ContainerRunner`](#containerrunner) reference below, which manages the container lifecycle.

**How it works:**

1. Builds a JSON payload with the prompt, wrapped agent ID, and model.
2. Invokes `ContainerRunner.run()`, which starts a container with `--network none` and `--rm`, writes the payload to stdin, and waits for JSON output on stdout.
3. Mount paths are validated against a configurable allowlist before the container is started.
4. Parses the sentinel-delimited output and returns an `AgentResult`.

**Constructor parameters:**

| Parameter              | Type              | Default      | Description                                       |
|------------------------|-------------------|--------------|---------------------------------------------------|
| `agent`                | `BaseAgent`       | --           | The wrapped agent to execute inside the container |
| `runner`               | `ContainerRunner` | --           | Container runner managing Docker lifecycle        |
| `engine`               | `InferenceEngine` | `None`       | Override engine (defaults to wrapped agent's)     |
| `model`                | `str`             | `""`         | Override model (defaults to wrapped agent's)      |
| `workspace`            | `str`             | `""`         | Working directory inside the container            |
| `mounts`               | `list[str]`       | `[]`         | Host paths to bind-mount (read-only)              |
| `secrets`              | `dict[str, str]`  | `{}`         | Injected into payload (not environment variables) |
| `bus`                  | `EventBus`        | `None`       | Event bus for telemetry                           |

```python
from openjarvis.sandbox import ContainerRunner, SandboxedAgent
from openjarvis.agents.simple import SimpleAgent

runner = ContainerRunner(
    image="openjarvis-sandbox:latest",
    timeout=60,
    mount_allowlist_path="/etc/openjarvis/mount_allowlist.json",
)
inner = SimpleAgent(engine, model="qwen3:8b")
agent = SandboxedAgent(
    agent=inner,
    runner=runner,
    mounts=["/home/user/data"],
)
result = agent.run("Summarize the CSV files in /home/user/data")
```

---

## ContainerRunner

`ContainerRunner` manages the Docker (or Podman) container lifecycle for sandboxed execution. It is used directly by `SandboxedAgent` but can also be used standalone.

**Constructor parameters:**

| Parameter              | Type   | Default                      | Description                                    |
|------------------------|--------|------------------------------|------------------------------------------------|
| `image`                | `str`  | `"openjarvis-sandbox:latest"`| Docker image to run                            |
| `timeout`              | `int`  | `300`                        | Max container execution time in seconds        |
| `mount_allowlist_path` | `str`  | `""`                         | Path to JSON mount-allowlist file              |
| `max_concurrent`       | `int`  | `5`                          | Max concurrent containers (informational)      |
| `runtime`              | `str`  | `"docker"`                   | Container runtime binary (`docker` or `podman`)|

**Mount allowlist format:**

```json title="mount_allowlist.json"
{
  "roots": [
    {"path": "/home/user/projects", "read_only": false},
    {"path": "/data/shared", "read_only": true}
  ],
  "blocked_patterns": [".ssh", ".env", "*.pem", "*.key"]
}
```

If `mount_allowlist_path` is not set, no root restriction is applied. Blocked patterns always include `.ssh`, `.env`, `*.pem`, `*.key`, credential files, and cloud config directories by default.

!!! warning "Docker required"
    `ContainerRunner` raises `RuntimeError` if the configured runtime (`docker` or `podman`) is not found on `PATH`.

---

## Agent Registration

Agents are registered via the `@AgentRegistry.register()` decorator. This makes them discoverable by name at runtime:

```python
from openjarvis.core.registry import AgentRegistry

# Check if an agent is registered
AgentRegistry.contains("orchestrator")  # True

# Get the agent class
agent_cls = AgentRegistry.get("orchestrator")

# List all registered agent keys
AgentRegistry.keys()
# ["simple", "orchestrator", "native_react", "react", "native_openhands",
#  "rlm", "openhands", "claude_code"]
```

---

## Using Agents

### Via CLI

```bash
# Simple agent
jarvis ask --agent simple "What is the capital of France?"

# Orchestrator with tools
jarvis ask --agent orchestrator --tools calculator,think "What is sqrt(256)?"

# NativeReActAgent
jarvis ask --agent native_react --tools calculator "What is 2+2?"

# ReAct alias (same as native_react)
jarvis ask --agent react --tools calculator,think "Solve step by step: 15% of 340"

# NativeOpenHandsAgent
jarvis ask --agent native_openhands --tools calculator,web_search "Summarize example.com"

# RLMAgent
jarvis ask --agent rlm "Summarize this long document"

# OpenHands SDK agent
jarvis ask --agent openhands "Fix the bug in test_utils.py"

# OpenClaw agent
jarvis ask --agent openclaw "Tell me a story"

# Claude Code agent (requires Node.js 22+ and ANTHROPIC_API_KEY)
jarvis ask --agent claude_code "Add docstrings to all functions in utils.py"
```

### Via Python SDK

```python
from openjarvis import Jarvis

j = Jarvis()

# Simple agent
response = j.ask("Hello", agent="simple")

# Orchestrator with tools
response = j.ask(
    "Calculate 15% of 340",
    agent="orchestrator",
    tools=["calculator"],
)

# NativeReActAgent with tools
response = j.ask(
    "What is sqrt(256)?",
    agent="native_react",
    tools=["calculator", "think"],
)

# Full result with tool details
result = j.ask_full(
    "What is the square root of 144?",
    agent="orchestrator",
    tools=["calculator", "think"],
)
print(result["content"])
print(result["turns"])
print(result["tool_results"])

j.close()
```

---

## Event Bus Integration

All agents publish events on the `EventBus` when a bus is provided:

| Event                   | When                                                |
|-------------------------|-----------------------------------------------------|
| `AGENT_TURN_START`      | At the beginning of a run (via `_emit_turn_start`)  |
| `AGENT_TURN_END`        | At the end of a run (via `_emit_turn_end`)          |
| `TOOL_CALL_START`       | Before each tool execution (`ToolUsingAgent` subclasses) |
| `TOOL_CALL_END`         | After each tool execution (`ToolUsingAgent` subclasses)  |

!!! info "Inference events"
    `INFERENCE_START` / `INFERENCE_END` events are published by the `InstrumentedEngine` wrapper, not by agents directly. This keeps telemetry opt-in and transparent to agent code.

These events enable the telemetry and trace systems to record detailed interaction data automatically.
