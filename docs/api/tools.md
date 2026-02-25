# Tools Module

The tools module implements the tool system used by agents for executing
actions such as calculations, memory search, file reading, and sub-model
queries. Each tool implements the `BaseTool` ABC and is registered via
`@ToolRegistry.register("name")`. The `ToolExecutor` dispatches tool calls
with JSON argument parsing, latency tracking, and event bus integration.

## Abstract Base Class

### BaseTool

::: openjarvis.tools._stubs.BaseTool
    options:
      show_source: true
      members_order: source

### ToolSpec

::: openjarvis.tools._stubs.ToolSpec
    options:
      show_source: true
      members_order: source

### ToolExecutor

::: openjarvis.tools._stubs.ToolExecutor
    options:
      show_source: true
      members_order: source

### build_tool_descriptions

::: openjarvis.tools._stubs.build_tool_descriptions
    options:
      show_source: true

---

## Built-in Tools

### CalculatorTool

::: openjarvis.tools.calculator.CalculatorTool
    options:
      show_source: true
      members_order: source

### safe_eval

::: openjarvis.tools.calculator.safe_eval
    options:
      show_source: true

### ThinkTool

::: openjarvis.tools.think.ThinkTool
    options:
      show_source: true
      members_order: source

### RetrievalTool

::: openjarvis.tools.retrieval.RetrievalTool
    options:
      show_source: true
      members_order: source

### LLMTool

::: openjarvis.tools.llm_tool.LLMTool
    options:
      show_source: true
      members_order: source

### FileReadTool

::: openjarvis.tools.file_read.FileReadTool
    options:
      show_source: true
      members_order: source

---

## Storage Tools

Tools that expose memory storage operations for use by agents via the tool
system. These wrap the memory backend methods as `BaseTool` implementations.

### MemoryStoreTool

::: openjarvis.tools.storage_tools.MemoryStoreTool
    options:
      show_source: true
      members_order: source

### MemoryRetrieveTool

::: openjarvis.tools.storage_tools.MemoryRetrieveTool
    options:
      show_source: true
      members_order: source

### MemorySearchTool

::: openjarvis.tools.storage_tools.MemorySearchTool
    options:
      show_source: true
      members_order: source

### MemoryIndexTool

::: openjarvis.tools.storage_tools.MemoryIndexTool
    options:
      show_source: true
      members_order: source

---

## Scheduler Tools

MCP tools for scheduling agent tasks for future or recurring execution. All five tools are in the `scheduler` category and require a `TaskScheduler` instance to be injected at `_scheduler`. For usage and schedule type examples see the [Scheduler Tools user guide](../user-guide/tools.md#scheduler-tools).

### ScheduleTaskTool

::: openjarvis.scheduler.tools.ScheduleTaskTool
    options:
      show_source: true
      members_order: source

### ListScheduledTasksTool

::: openjarvis.scheduler.tools.ListScheduledTasksTool
    options:
      show_source: true
      members_order: source

### PauseScheduledTaskTool

::: openjarvis.scheduler.tools.PauseScheduledTaskTool
    options:
      show_source: true
      members_order: source

### ResumeScheduledTaskTool

::: openjarvis.scheduler.tools.ResumeScheduledTaskTool
    options:
      show_source: true
      members_order: source

### CancelScheduledTaskTool

::: openjarvis.scheduler.tools.CancelScheduledTaskTool
    options:
      show_source: true
      members_order: source

---

## MCP Adapter

Integration with the Model Context Protocol (MCP). The MCP adapter exposes
OpenJarvis tools as MCP-compatible resources and allows consuming external
MCP tool providers. Supports MCP protocol version 2025-11-25.

### MCPToolAdapter

::: openjarvis.tools.mcp_adapter.MCPToolAdapter
    options:
      show_source: true
      members_order: source

### MCPToolProvider

::: openjarvis.tools.mcp_adapter.MCPToolProvider
    options:
      show_source: true
      members_order: source
