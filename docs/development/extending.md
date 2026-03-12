# Extending OpenJarvis

OpenJarvis is designed to be extended through its registry pattern. Every
major subsystem defines an abstract base class (ABC) and uses a typed registry
for runtime discovery. To add a new component, implement the ABC, decorate
it with the registry, and import it in the module's `__init__.py`.

This guide provides complete, working code examples for each extension point.

---

## Adding a New Inference Engine

Inference engines connect OpenJarvis to an LLM runtime. All engines implement
the `InferenceEngine` ABC defined in `engine/_stubs.py`.

### Step 1: Create the Engine Module

Create `src/openjarvis/engine/my_engine.py`:

```python
"""My custom inference engine backend."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Any, Dict, List

import httpx

from openjarvis.core.registry import EngineRegistry
from openjarvis.core.types import Message
from openjarvis.engine._base import (
    EngineConnectionError,
    InferenceEngine,
    messages_to_dicts,
)


@EngineRegistry.register("my_engine")  # (1)!
class MyEngine(InferenceEngine):
    """Custom inference engine backend."""

    engine_id = "my_engine"  # (2)!

    def __init__(
        self,
        host: str = "http://localhost:9000",
        *,
        timeout: float = 120.0,
    ) -> None:
        self._host = host.rstrip("/")
        self._client = httpx.Client(base_url=self._host, timeout=timeout)

    def generate(
        self,
        messages: Sequence[Message],
        *,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Synchronous completion."""
        payload = {
            "model": model,
            "messages": messages_to_dicts(messages),  # (3)!
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        # Pass tools if provided
        tools = kwargs.get("tools")
        if tools:
            payload["tools"] = tools

        try:
            resp = self._client.post("/v1/chat/completions", json=payload)
            resp.raise_for_status()
        except (httpx.ConnectError, httpx.TimeoutException) as exc:
            raise EngineConnectionError(
                f"Engine not reachable at {self._host}"
            ) from exc

        data = resp.json()
        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {})
        usage = data.get("usage", {})

        result: Dict[str, Any] = {
            "content": message.get("content", ""),
            "usage": {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            },
            "model": data.get("model", model),
            "finish_reason": choice.get("finish_reason", "stop"),
        }

        # Extract tool calls if present
        raw_tool_calls = message.get("tool_calls", [])
        if raw_tool_calls:
            result["tool_calls"] = [
                {
                    "id": tc.get("id", f"call_{i}"),
                    "name": tc["function"]["name"],
                    "arguments": tc["function"]["arguments"],
                }
                for i, tc in enumerate(raw_tool_calls)
            ]
        return result

    async def stream(
        self,
        messages: Sequence[Message],
        *,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Yield token strings as they are generated."""
        # Implement SSE or WebSocket streaming for your engine
        result = self.generate(
            messages, model=model, temperature=temperature,
            max_tokens=max_tokens, **kwargs,
        )
        yield result.get("content", "")

    def list_models(self) -> List[str]:
        """Return identifiers of models available on this engine."""
        try:
            resp = self._client.get("/v1/models")
            resp.raise_for_status()
            data = resp.json()
            return [m["id"] for m in data.get("data", [])]
        except Exception:
            return []

    def health(self) -> bool:
        """Return True when the engine is reachable and healthy."""
        try:
            resp = self._client.get("/health", timeout=2.0)
            return resp.status_code == 200
        except Exception:
            return False
```

1. The `@EngineRegistry.register("my_engine")` decorator makes this engine
   discoverable by key at runtime.
2. The `engine_id` class attribute is used in telemetry and benchmark results.
3. `messages_to_dicts()` converts `Message` objects to OpenAI-format dicts.

### Step 2: Register in `__init__.py`

Add your engine import to `src/openjarvis/engine/__init__.py`:

```python
import openjarvis.engine.my_engine  # noqa: F401
```

If your engine requires optional dependencies, wrap the import:

```python
try:
    import openjarvis.engine.my_engine  # noqa: F401
except ImportError:
    pass
```

### Step 3: Add Optional Dependencies

If your engine needs extra packages, add them to `pyproject.toml`:

```toml
[project.optional-dependencies]
inference-myengine = [
    "my-engine-sdk>=1.0",
]
```

### Required ABC Methods

| Method | Signature | Returns | Description |
|---|---|---|---|
| `generate` | `(messages, *, model, temperature, max_tokens, **kwargs)` | `Dict[str, Any]` | Synchronous completion with `content` and `usage` keys |
| `stream` | `(messages, *, model, temperature, max_tokens, **kwargs)` | `AsyncIterator[str]` | Yields token strings as they are generated |
| `list_models` | `()` | `List[str]` | Model identifiers available on this engine |
| `health` | `()` | `bool` | `True` when the engine is reachable |

The `generate` return dict must include at minimum:

```python
{
    "content": "The response text",
    "usage": {
        "prompt_tokens": 10,
        "completion_tokens": 20,
        "total_tokens": 30,
    },
    "model": "model-name",
    "finish_reason": "stop",  # or "tool_calls"
}
```

!!! tip "Tool call support"
    If your engine supports tool/function calling, include a `"tool_calls"`
    key in the return dict. Each tool call should have `id`, `name`, and
    `arguments` (JSON string) keys.

---

## Adding a New Memory Backend

Memory backends provide persistent, searchable storage. All backends implement
the `MemoryBackend` ABC defined in `tools/storage/_stubs.py` (previously `memory/_stubs.py`).

### Complete Example

Create `src/openjarvis/tools/storage/my_backend.py`:

```python
"""Custom memory backend example."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from openjarvis.core.registry import MemoryRegistry
from openjarvis.tools.storage._stubs import MemoryBackend, RetrievalResult


@MemoryRegistry.register("my_backend")
class MyMemoryBackend(MemoryBackend):
    """Custom memory backend implementation."""

    backend_id = "my_backend"

    def __init__(self, **kwargs: Any) -> None:
        # Initialize your storage (database, index, etc.)
        self._store: Dict[str, Dict[str, Any]] = {}

    def store(
        self,
        content: str,
        *,
        source: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Persist content and return a unique document id."""
        import uuid

        doc_id = uuid.uuid4().hex
        self._store[doc_id] = {
            "content": content,
            "source": source,
            "metadata": metadata or {},
        }
        return doc_id

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 5,
        **kwargs: Any,
    ) -> List[RetrievalResult]:
        """Search for query and return the top-k results."""
        results: List[RetrievalResult] = []
        for doc_id, doc in self._store.items():
            # Implement your search/ranking logic here
            if query.lower() in doc["content"].lower():
                results.append(RetrievalResult(
                    content=doc["content"],
                    score=1.0,
                    source=doc["source"],
                    metadata=doc["metadata"],
                ))
        return results[:top_k]

    def delete(self, doc_id: str) -> bool:
        """Delete a document by id. Return True if it existed."""
        return self._store.pop(doc_id, None) is not None

    def clear(self) -> None:
        """Remove all stored documents."""
        self._store.clear()
```

### Register in `__init__.py`

Add to `src/openjarvis/tools/storage/__init__.py`:

```python
try:
    import openjarvis.tools.storage.my_backend  # noqa: F401
except ImportError:
    pass
```

!!! note "Backward compatibility"
    The old `from openjarvis.memory._stubs import MemoryBackend` import path still works via backward-compatibility shims, but new code should use `openjarvis.tools.storage._stubs`.

### Required ABC Methods

| Method | Signature | Returns | Description |
|---|---|---|---|
| `store` | `(content, *, source, metadata)` | `str` | Persist content, return document ID |
| `retrieve` | `(query, *, top_k, **kwargs)` | `List[RetrievalResult]` | Search and return ranked results |
| `delete` | `(doc_id)` | `bool` | Delete by ID, return whether it existed |
| `clear` | `()` | `None` | Remove all stored documents |

The `RetrievalResult` dataclass has these fields:

```python
@dataclass(slots=True)
class RetrievalResult:
    content: str                          # The retrieved text
    score: float = 0.0                    # Relevance score
    source: str = ""                      # Source identifier
    metadata: Dict[str, Any] = field(default_factory=dict)
```

---

## Adding a New Agent

Agents implement the logic for handling queries, calling tools, and managing
multi-turn interactions. There are two paths depending on whether your agent
uses tools:

- **Path A: Non-tool agent** -- Extend `BaseAgent` directly
- **Path B: Tool-using agent** -- Extend `ToolUsingAgent` (which sets `accepts_tools = True` and provides a `ToolExecutor`)

### Path A: Non-tool Agent (extending BaseAgent)

Create `src/openjarvis/agents/my_agent.py`:

```python
"""Custom agent implementation — single-turn, no tools."""

from __future__ import annotations

from typing import Any, Optional

from openjarvis.agents._stubs import AgentContext, AgentResult, BaseAgent
from openjarvis.core.registry import AgentRegistry
from openjarvis.engine._stubs import InferenceEngine


@AgentRegistry.register("my_agent")
class MyAgent(BaseAgent):
    """Custom agent with specialized behavior."""

    agent_id = "my_agent"

    def run(
        self,
        input: str,
        context: Optional[AgentContext] = None,
        **kwargs: Any,
    ) -> AgentResult:
        """Execute the agent on input and return an AgentResult."""
        # Use BaseAgent helpers instead of manual event bus code
        self._emit_turn_start(input)

        # Build messages from context + user input (with optional system prompt)
        messages = self._build_messages(
            input, context,
            system_prompt="You are a helpful assistant with specialized knowledge.",
        )

        # Call engine.generate() with stored defaults (model, temperature, max_tokens)
        result = self._generate(messages)
        content = self._strip_think_tags(result.get("content", ""))

        self._emit_turn_end(turns=1)
        return AgentResult(content=content, turns=1)
```

!!! tip "BaseAgent helpers"
    `BaseAgent` provides these concrete helpers so you don't need to manually
    manage the event bus or engine calls:

    | Helper | Purpose |
    |--------|---------|
    | `_emit_turn_start(input)` | Publish `AGENT_TURN_START` |
    | `_emit_turn_end(**data)` | Publish `AGENT_TURN_END` |
    | `_build_messages(input, context, *, system_prompt)` | Assemble message list |
    | `_generate(messages, **kwargs)` | Call engine with stored defaults |
    | `_strip_think_tags(text)` | Remove `<think>` blocks |
    | `_max_turns_result(tool_results, turns, content)` | Standard max-turns result |

### Path B: Tool-using Agent (extending ToolUsingAgent)

Create `src/openjarvis/agents/my_tool_agent.py`:

```python
"""Custom tool-using agent with a multi-turn loop."""

from __future__ import annotations

from typing import Any, List, Optional

from openjarvis.agents._stubs import AgentContext, AgentResult, ToolUsingAgent
from openjarvis.core.events import EventBus
from openjarvis.core.registry import AgentRegistry
from openjarvis.core.types import ToolCall, ToolResult
from openjarvis.engine._stubs import InferenceEngine
from openjarvis.tools._stubs import BaseTool


@AgentRegistry.register("my_tool_agent")
class MyToolAgent(ToolUsingAgent):
    """Custom agent with tool-calling loop."""

    agent_id = "my_tool_agent"

    def __init__(
        self,
        engine: InferenceEngine,
        model: str,
        *,
        tools: Optional[List[BaseTool]] = None,
        bus: Optional[EventBus] = None,
        max_turns: int = 10,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> None:
        super().__init__(
            engine, model, tools=tools, bus=bus,
            max_turns=max_turns, temperature=temperature,
            max_tokens=max_tokens,
        )

    def run(
        self,
        input: str,
        context: Optional[AgentContext] = None,
        **kwargs: Any,
    ) -> AgentResult:
        self._emit_turn_start(input)

        messages = self._build_messages(input, context)
        tools_spec = self._executor.get_openai_tools()
        all_tool_results: list[ToolResult] = []
        turns = 0

        for _ in range(self._max_turns):
            turns += 1
            result = self._generate(messages, tools=tools_spec)
            content = result.get("content", "")
            tool_calls = result.get("tool_calls", [])

            if not tool_calls:
                self._emit_turn_end(turns=turns)
                return AgentResult(
                    content=content,
                    tool_results=all_tool_results,
                    turns=turns,
                )

            # Execute each tool call
            for tc in tool_calls:
                call = ToolCall(
                    id=tc.get("id", f"call_{turns}"),
                    name=tc["name"],
                    arguments=tc["arguments"],
                )
                tr = self._executor.execute(call)
                all_tool_results.append(tr)

        # Max turns exceeded — use the standard helper
        return self._max_turns_result(all_tool_results, turns)
```

!!! info "What ToolUsingAgent adds"
    `ToolUsingAgent` extends `BaseAgent` with:

    - **`accepts_tools = True`** — enables `--tools` in CLI and `tools=` in SDK
    - **`self._executor`** — a `ToolExecutor` initialized from the provided tools
    - **`self._tools`** — the raw list of `BaseTool` instances
    - **`self._max_turns`** — configurable loop iteration limit (default: 10)

### Register in `__init__.py`

Add to `src/openjarvis/agents/__init__.py`:

```python
try:
    import openjarvis.agents.my_agent  # noqa: F401
except ImportError:
    pass
```

### Key Types

=== "AgentContext"

    ```python
    @dataclass(slots=True)
    class AgentContext:
        conversation: Conversation = field(default_factory=Conversation)
        tools: List[str] = field(default_factory=list)
        memory_results: List[Any] = field(default_factory=list)
        metadata: Dict[str, Any] = field(default_factory=dict)
    ```

=== "AgentResult"

    ```python
    @dataclass(slots=True)
    class AgentResult:
        content: str
        tool_results: List[ToolResult] = field(default_factory=list)
        turns: int = 0
        metadata: Dict[str, Any] = field(default_factory=dict)
    ```

---

## Adding a New Tool

Tools are callable capabilities that agents can invoke during multi-turn
reasoning. All tools implement the `BaseTool` ABC from `tools/_stubs.py`.

### Complete Example

Create `src/openjarvis/tools/my_tool.py`:

```python
"""Custom tool implementation."""

from __future__ import annotations

from typing import Any

from openjarvis.core.registry import ToolRegistry
from openjarvis.core.types import ToolResult
from openjarvis.tools._stubs import BaseTool, ToolSpec


@ToolRegistry.register("my_tool")
class MyTool(BaseTool):
    """A custom tool that does something useful."""

    tool_id = "my_tool"

    @property
    def spec(self) -> ToolSpec:
        """Return the tool specification."""
        return ToolSpec(
            name="my_tool",
            description="Does something useful with the provided input.",
            parameters={  # (1)!
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The input to process",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
            category="utility",
            cost_estimate=0.001,       # Estimated cost in USD per call
            latency_estimate=0.5,      # Estimated latency in seconds
            requires_confirmation=False,
        )

    def execute(self, **params: Any) -> ToolResult:
        """Execute the tool with the given parameters."""
        query = params.get("query", "")
        max_results = params.get("max_results", 5)

        if not query:
            return ToolResult(
                tool_name="my_tool",
                content="No query provided.",
                success=False,
            )

        try:
            # Your tool logic here
            result_text = f"Processed '{query}' (max_results={max_results})"

            return ToolResult(
                tool_name="my_tool",
                content=result_text,
                success=True,
            )
        except Exception as exc:
            return ToolResult(
                tool_name="my_tool",
                content=f"Error: {exc}",
                success=False,
            )
```

1. The `parameters` dict follows the [JSON Schema](https://json-schema.org/)
   format used by OpenAI function calling. The `ToolExecutor` will parse
   incoming JSON arguments and pass them as keyword arguments to `execute()`.

### Register in `__init__.py`

Add to `src/openjarvis/tools/__init__.py`:

```python
try:
    import openjarvis.tools.my_tool  # noqa: F401
except ImportError:
    pass
```

### How Tools Are Invoked

The `ToolExecutor` handles the dispatch loop:

1. The agent's LLM generates a `tool_calls` response with tool name and
   JSON arguments
2. `ToolExecutor.execute()` parses the JSON arguments
3. The matching tool's `execute(**params)` is called
4. The `ToolResult` is returned to the agent for the next turn

```python
from openjarvis.tools._stubs import ToolExecutor

executor = ToolExecutor(
    tools=[MyTool()],
    bus=event_bus,  # Optional — enables TOOL_CALL_START/END events
)

# Dispatch a tool call
from openjarvis.core.types import ToolCall

call = ToolCall(id="call_1", name="my_tool", arguments='{"query": "test"}')
result = executor.execute(call)
```

The `to_openai_function()` method converts a tool's spec to OpenAI function
calling format, which is sent to the LLM alongside the conversation:

```python
tool = MyTool()
openai_format = tool.to_openai_function()
# {
#     "type": "function",
#     "function": {
#         "name": "my_tool",
#         "description": "Does something useful...",
#         "parameters": { ... }
#     }
# }
```

---

## Adding a New Benchmark

Benchmarks measure engine performance. All benchmarks implement the
`BaseBenchmark` ABC from `bench/_stubs.py` and use the `ensure_registered()`
pattern for lazy registration.

### Complete Example

Create `src/openjarvis/bench/my_benchmark.py`:

```python
"""Custom benchmark — measures time to first token."""

from __future__ import annotations

import time

from openjarvis.bench._stubs import BaseBenchmark, BenchmarkResult
from openjarvis.core.registry import BenchmarkRegistry
from openjarvis.core.types import Message, Role
from openjarvis.engine._stubs import InferenceEngine


class TTFTBenchmark(BaseBenchmark):
    """Measures time-to-first-token across multiple samples."""

    @property
    def name(self) -> str:
        return "ttft"

    @property
    def description(self) -> str:
        return "Measures time-to-first-token latency"

    def run(
        self,
        engine: InferenceEngine,
        model: str,
        *,
        num_samples: int = 10,
    ) -> BenchmarkResult:
        ttft_values: list[float] = []
        errors = 0

        for _ in range(num_samples):
            messages = [Message(role=Role.USER, content="Hello")]
            t0 = time.time()
            try:
                engine.generate(messages, model=model)
                ttft_values.append(time.time() - t0)
            except Exception:
                errors += 1

        if not ttft_values:
            return BenchmarkResult(
                benchmark_name=self.name,
                model=model,
                engine=engine.engine_id,
                metrics={},
                samples=num_samples,
                errors=errors,
            )

        return BenchmarkResult(
            benchmark_name=self.name,
            model=model,
            engine=engine.engine_id,
            metrics={
                "mean_ttft": sum(ttft_values) / len(ttft_values),
                "min_ttft": min(ttft_values),
                "max_ttft": max(ttft_values),
            },
            samples=num_samples,
            errors=errors,
        )


def ensure_registered() -> None:  # (1)!
    """Register the TTFT benchmark if not already present."""
    if not BenchmarkRegistry.contains("ttft"):
        BenchmarkRegistry.register_value("ttft", TTFTBenchmark)
```

1. The `ensure_registered()` function uses `contains()` before
   `register_value()` so it can be called multiple times safely. This is
   required because tests clear all registries between runs.

### Register in `__init__.py`

Update `src/openjarvis/bench/__init__.py` to call `ensure_registered()`:

```python
from openjarvis.bench.my_benchmark import ensure_registered as _reg_ttft
_reg_ttft()
```

### BenchmarkResult Fields

```python
@dataclass(slots=True)
class BenchmarkResult:
    benchmark_name: str                    # e.g. "latency", "throughput"
    model: str                             # Model identifier
    engine: str                            # Engine identifier
    metrics: Dict[str, float] = ...        # Measured values
    metadata: Dict[str, Any] = ...         # Extra info
    samples: int = 0                       # Number of samples run
    errors: int = 0                        # Number of failed samples
```

---

## Adding a New Router Policy

Router policies determine which model handles a given query. All policies
implement the `RouterPolicy` ABC from `learning/_stubs.py`. The
`RoutingContext` dataclass is defined in `core/types.py`.

### Complete Example

Create `src/openjarvis/learning/my_policy.py`:

```python
"""Custom router policy — selects model based on query length."""

from __future__ import annotations

from typing import List, Optional

from openjarvis.core.registry import RouterPolicyRegistry
from openjarvis.core.types import RoutingContext
from openjarvis.learning._stubs import RouterPolicy


class QueryLengthPolicy(RouterPolicy):
    """Routes queries to models based on query length.

    Short queries go to a fast, small model. Long or complex queries
    go to a larger, more capable model.
    """

    def __init__(
        self,
        available_models: Optional[List[str]] = None,
        *,
        default_model: str = "",
        fallback_model: str = "",
        short_threshold: int = 100,
        long_threshold: int = 500,
    ) -> None:
        self._available = available_models or []
        self._default = default_model
        self._fallback = fallback_model
        self._short_threshold = short_threshold
        self._long_threshold = long_threshold

    def select_model(self, context: RoutingContext) -> str:
        """Return the model registry key best suited for this context."""
        available = self._available

        if not available:
            return self._default or self._fallback or ""

        if context.query_length < self._short_threshold:
            # Prefer the first (presumably smallest) available model
            return available[0]
        elif context.query_length > self._long_threshold:
            # Prefer the last (presumably largest) available model
            return available[-1]

        # Default to configured model
        if self._default and self._default in available:
            return self._default
        return available[0]


def ensure_registered() -> None:
    """Register QueryLengthPolicy if not already present."""
    if not RouterPolicyRegistry.contains("query_length"):
        RouterPolicyRegistry.register_value("query_length", QueryLengthPolicy)


ensure_registered()
```

### Register in `__init__.py`

Update `src/openjarvis/learning/__init__.py`:

```python
from openjarvis.learning.my_policy import ensure_registered as _reg_ql
_reg_ql()
```

### Using Your Policy

Once registered, your policy can be selected via the config file or CLI:

=== "Config (TOML)"

    ```toml
    [learning.routing]
    policy = "query_length"
    ```

=== "CLI"

    ```bash
    uv run jarvis ask --router query_length "Hello"
    ```

### The RoutingContext

The `RoutingContext` dataclass provides all the information a router needs:

```python
@dataclass(slots=True)
class RoutingContext:
    query: str = ""
    query_length: int = 0
    has_code: bool = False
    has_math: bool = False
    language: str = "en"
    urgency: float = 0.5      # 0 = low priority, 1 = real-time
    metadata: Dict[str, Any] = field(default_factory=dict)
```

The `build_routing_context()` helper in `learning/router.py` populates
this from a raw query string, detecting code and math patterns automatically.

---

## Summary

| Component | ABC | Registry | Key location |
|---|---|---|---|
| Inference Engine | `InferenceEngine` | `EngineRegistry` | `engine/_stubs.py` |
| Memory Backend | `MemoryBackend` | `MemoryRegistry` | `tools/storage/_stubs.py` |
| Agent | `BaseAgent` | `AgentRegistry` | `agents/_stubs.py` |
| Tool | `BaseTool` | `ToolRegistry` | `tools/_stubs.py` |
| Benchmark | `BaseBenchmark` | `BenchmarkRegistry` | `bench/_stubs.py` |
| Router Policy | `RouterPolicy` | `RouterPolicyRegistry` | `learning/_stubs.py` |
| Learning Policy | `LearningPolicy` | `LearningRegistry` | `learning/_stubs.py` |

The general pattern for all extension points:

1. Implement the ABC in a new module
2. Decorate the class with `@XRegistry.register("key")` or use
   `ensure_registered()` for lazy registration
3. Import the module in the package's `__init__.py` (with `try/except
   ImportError` if optional deps are involved)
4. Add tests in `tests/<module>/`
5. Add optional dependencies to `pyproject.toml` if needed
