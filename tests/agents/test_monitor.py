"""Tests for the MonitorAgent."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from openjarvis.agents._stubs import AgentContext
from openjarvis.agents.monitor import MonitorAgent
from openjarvis.core.events import EventBus, EventType
from openjarvis.core.types import Conversation, Message, Role, ToolResult
from openjarvis.tools._stubs import BaseTool, ToolSpec

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _CalculatorStub(BaseTool):
    tool_id = "calculator"

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="calculator",
            description="Math calculator.",
            parameters={
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        )

    def execute(self, **params) -> ToolResult:
        expr = params.get("expression", "0")
        try:
            val = eval(expr)  # noqa: S307
        except Exception as e:
            return ToolResult(tool_name="calculator", content=str(e), success=False)
        return ToolResult(tool_name="calculator", content=str(val), success=True)


class _CodeInterpreterStub(BaseTool):
    tool_id = "code_interpreter"

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="code_interpreter",
            description="Execute Python code.",
            parameters={
                "type": "object",
                "properties": {"code": {"type": "string"}},
                "required": ["code"],
            },
        )

    def execute(self, **params) -> ToolResult:
        code = params.get("code", "")
        # Simple stub: just return the code as output
        return ToolResult(
            tool_name="code_interpreter",
            content=f"executed: {code[:50]}",
            success=True,
        )


class _ThinkStub(BaseTool):
    tool_id = "think"

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="think",
            description="Thinking tool.",
            parameters={
                "type": "object",
                "properties": {"thought": {"type": "string"}},
            },
        )

    def execute(self, **params) -> ToolResult:
        return ToolResult(
            tool_name="think",
            content=params.get("thought", ""),
            success=True,
        )


def _make_engine_no_tools(content: str = "Final answer.") -> MagicMock:
    """Engine that never returns tool calls."""
    engine = MagicMock()
    engine.engine_id = "mock"
    engine.generate.return_value = {
        "content": content,
        "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
        "model": "test-model",
        "finish_reason": "stop",
    }
    return engine


def _make_engine_with_tool_call(
    tool_name: str = "calculator",
    arguments: str = '{"expression":"2+2"}',
    tool_call_id: str = "call_1",
    final_content: str = "The answer is 4.",
) -> MagicMock:
    """Engine that returns one tool call then a final answer."""
    engine = MagicMock()
    engine.engine_id = "mock"
    engine.generate.side_effect = [
        {
            "content": "",
            "tool_calls": [
                {"id": tool_call_id, "name": tool_name, "arguments": arguments}
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
            "model": "test-model",
            "finish_reason": "tool_calls",
        },
        {
            "content": final_content,
            "usage": {"prompt_tokens": 15, "completion_tokens": 5, "total_tokens": 20},
            "model": "test-model",
            "finish_reason": "stop",
        },
    ]
    return engine


def _make_engine_with_code(
    code: str = "print(2+2)",
    final_content: str = "The result is 4.",
) -> MagicMock:
    """Engine that returns a Python code block, then a final answer."""
    engine = MagicMock()
    engine.engine_id = "mock"
    code_response = f"Let me calculate:\n```python\n{code}\n```"
    engine.generate.side_effect = [
        {
            "content": code_response,
            "usage": {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15},
            "model": "test-model",
            "finish_reason": "stop",
        },
        {
            "content": final_content,
            "usage": {"prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25},
            "model": "test-model",
            "finish_reason": "stop",
        },
    ]
    return engine


def _make_engine_interleaved() -> MagicMock:
    """Engine that interleaves tool calls and code blocks."""
    engine = MagicMock()
    engine.engine_id = "mock"
    engine.generate.side_effect = [
        # Turn 1: tool call
        {
            "content": "",
            "tool_calls": [
                {"id": "c1", "name": "calculator", "arguments": '{"expression":"2+2"}'}
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
            "model": "test-model",
            "finish_reason": "tool_calls",
        },
        # Turn 2: code block
        {
            "content": "Now processing:\n```python\nresult = 4 * 3\nprint(result)\n```",
            "usage": {"prompt_tokens": 15, "completion_tokens": 10, "total_tokens": 25},
            "model": "test-model",
            "finish_reason": "stop",
        },
        # Turn 3: final answer
        {
            "content": "Done. 2+2=4, 4*3=12",
            "usage": {"prompt_tokens": 30, "completion_tokens": 5, "total_tokens": 35},
            "model": "test-model",
            "finish_reason": "stop",
        },
    ]
    return engine


# ---------------------------------------------------------------------------
# Tests — Initialization
# ---------------------------------------------------------------------------


class TestMonitorInit:
    def test_agent_id(self):
        engine = _make_engine_no_tools()
        agent = MonitorAgent(engine, "test-model")
        assert agent.agent_id == "monitor"

    def test_accepts_tools(self):
        engine = _make_engine_no_tools()
        agent = MonitorAgent(engine, "test-model")
        assert agent.accepts_tools is True

    def test_default_parameters(self):
        engine = _make_engine_no_tools()
        agent = MonitorAgent(engine, "test-model")
        assert agent._max_turns == 25
        assert agent._temperature == 0.3
        assert agent._max_tokens == 4096

    def test_custom_parameters(self):
        engine = _make_engine_no_tools()
        agent = MonitorAgent(
            engine, "test-model",
            max_turns=10,
            temperature=0.5,
            max_tokens=2048,
        )
        assert agent._max_turns == 10
        assert agent._temperature == 0.5
        assert agent._max_tokens == 2048

    def test_operator_params(self):
        engine = _make_engine_no_tools()
        mem = MagicMock()
        sess = MagicMock()
        agent = MonitorAgent(
            engine, "test-model",
            operator_id="test-op",
            memory_backend=mem,
            session_store=sess,
        )
        assert agent._operator_id == "test-op"
        assert agent._memory_backend is mem
        assert agent._session_store is sess

    def test_code_execution_default_enabled(self):
        engine = _make_engine_no_tools()
        agent = MonitorAgent(engine, "test-model")
        assert agent._enable_code_execution is True

    def test_code_execution_disabled(self):
        engine = _make_engine_no_tools()
        agent = MonitorAgent(engine, "test-model", enable_code_execution=False)
        assert agent._enable_code_execution is False

    def test_custom_system_prompt(self):
        engine = _make_engine_no_tools()
        agent = MonitorAgent(engine, "test-model", system_prompt="Custom prompt")
        assert agent._system_prompt == "Custom prompt"


# ---------------------------------------------------------------------------
# Tests — Tool calling (OrchestratorAgent pattern)
# ---------------------------------------------------------------------------


class TestMonitorToolCalling:
    def test_no_tools_single_turn(self):
        engine = _make_engine_no_tools("Hello!")
        agent = MonitorAgent(engine, "test-model")
        result = agent.run("Hello")
        assert result.content == "Hello!"
        assert result.turns == 1
        assert result.tool_results == []

    def test_single_tool_call(self):
        engine = _make_engine_with_tool_call()
        agent = MonitorAgent(
            engine, "test-model", tools=[_CalculatorStub()],
        )
        result = agent.run("What is 2+2?")
        assert result.content == "The answer is 4."
        assert result.turns == 2
        assert len(result.tool_results) == 1
        assert result.tool_results[0].tool_name == "calculator"
        assert result.tool_results[0].content == "4"

    def test_tools_passed_to_engine(self):
        engine = _make_engine_no_tools()
        agent = MonitorAgent(
            engine, "test-model", tools=[_CalculatorStub()],
        )
        agent.run("Hello")
        call_kwargs = engine.generate.call_args[1]
        assert "tools" in call_kwargs

    def test_no_tools_no_tools_kwarg(self):
        engine = _make_engine_no_tools()
        agent = MonitorAgent(engine, "test-model")
        agent.run("Hello")
        call_kwargs = engine.generate.call_args[1]
        assert "tools" not in call_kwargs

    def test_unknown_tool(self):
        engine = _make_engine_with_tool_call(
            tool_name="unknown", arguments="{}",
            final_content="Handled.",
        )
        agent = MonitorAgent(
            engine, "test-model", tools=[_CalculatorStub()],
        )
        result = agent.run("Use unknown tool")
        assert result.content == "Handled."
        assert len(result.tool_results) == 1
        assert result.tool_results[0].success is False

    def test_max_turns_exceeded(self):
        engine = MagicMock()
        engine.engine_id = "mock"
        engine.generate.return_value = {
            "content": "",
            "tool_calls": [
                {"id": "c1", "name": "calculator", "arguments": '{"expression":"1+1"}'}
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
            "model": "test-model",
            "finish_reason": "tool_calls",
        }
        agent = MonitorAgent(
            engine, "test-model",
            tools=[_CalculatorStub()],
            max_turns=3,
        )
        result = agent.run("Loop forever")
        assert result.turns == 3
        assert result.metadata.get("max_turns_exceeded") is True

    def test_temperature_passthrough(self):
        engine = _make_engine_no_tools()
        agent = MonitorAgent(engine, "test-model", temperature=0.1)
        agent.run("Hello")
        call_kwargs = engine.generate.call_args[1]
        assert call_kwargs["temperature"] == 0.1


# ---------------------------------------------------------------------------
# Tests — Code execution (RLM pattern)
# ---------------------------------------------------------------------------


class TestMonitorCodeExecution:
    def test_code_block_detected_and_executed(self):
        engine = _make_engine_with_code()
        agent = MonitorAgent(
            engine, "test-model",
            tools=[_CodeInterpreterStub()],
        )
        result = agent.run("Calculate 2+2")
        assert result.turns == 2
        # Should have a code_interpreter tool result
        code_results = [
            r for r in result.tool_results
            if r.tool_name == "code_interpreter"
        ]
        assert len(code_results) == 1

    def test_code_execution_disabled(self):
        """When code execution is disabled, code blocks are ignored."""
        engine = _make_engine_with_code(
            code="print(42)",
            final_content="Let me calculate:\n```python\nprint(42)\n```",
        )
        # Override side_effect to return content with code on first call
        engine.generate.side_effect = [
            {
                "content": "```python\nprint(42)\n```",
                "usage": {
                    "prompt_tokens": 5,
                    "completion_tokens": 5,
                    "total_tokens": 10,
                },
                "model": "test-model",
                "finish_reason": "stop",
            },
        ]
        agent = MonitorAgent(
            engine, "test-model",
            enable_code_execution=False,
        )
        result = agent.run("Calculate")
        # Should return the content as-is since code execution is disabled
        assert result.turns == 1
        assert "```python" in result.content

    def test_extract_code_python_block(self):
        text = "Here:\n```python\nprint(42)\n```\nDone."
        assert MonitorAgent._extract_code(text) == "print(42)"

    def test_extract_code_bare_block(self):
        text = "Here:\n```\nprint(42)\n```\nDone."
        assert MonitorAgent._extract_code(text) == "print(42)"

    def test_extract_code_no_block(self):
        text = "Just plain text."
        assert MonitorAgent._extract_code(text) is None

    def test_execute_code_uses_tool(self):
        engine = _make_engine_no_tools()
        stub = _CodeInterpreterStub()
        agent = MonitorAgent(engine, "test-model", tools=[stub])
        output = agent._execute_code("print(1)")
        assert "executed:" in output

    def test_execute_code_fallback(self):
        """When code_interpreter is not in tools, falls back to import."""
        engine = _make_engine_no_tools()
        agent = MonitorAgent(engine, "test-model", tools=[_CalculatorStub()])
        # Patch the fallback import to avoid actual subprocess
        mock_tool = MagicMock()
        mock_tool.execute.return_value = ToolResult(
            tool_name="code_interpreter",
            content="fallback output",
            success=True,
        )
        with patch(
            "openjarvis.tools.code_interpreter.CodeInterpreterTool",
            return_value=mock_tool,
        ):
            output = agent._execute_code("print(1)")
        assert output == "fallback output"


# ---------------------------------------------------------------------------
# Tests — Interleaved tool + code execution
# ---------------------------------------------------------------------------


class TestMonitorHybridLoop:
    def test_interleaved_tool_and_code(self):
        engine = _make_engine_interleaved()
        agent = MonitorAgent(
            engine, "test-model",
            tools=[_CalculatorStub(), _CodeInterpreterStub()],
        )
        result = agent.run("Calculate and process")
        assert result.turns == 3
        assert result.content == "Done. 2+2=4, 4*3=12"
        # Should have both tool and code results
        assert len(result.tool_results) == 2
        tool_names = [r.tool_name for r in result.tool_results]
        assert "calculator" in tool_names
        assert "code_interpreter" in tool_names

    def test_messages_accumulate_with_tool(self):
        engine = _make_engine_with_tool_call()
        agent = MonitorAgent(
            engine, "test-model", tools=[_CalculatorStub()],
        )
        agent.run("What is 2+2?")
        second_call = engine.generate.call_args_list[1]
        messages = second_call[0][0]
        roles = [m.role for m in messages]
        assert Role.ASSISTANT in roles
        assert Role.TOOL in roles

    def test_messages_accumulate_with_code(self):
        engine = _make_engine_with_code()
        agent = MonitorAgent(
            engine, "test-model", tools=[_CodeInterpreterStub()],
        )
        agent.run("Calculate")
        second_call = engine.generate.call_args_list[1]
        messages = second_call[0][0]
        # Code results come back as USER messages
        user_msgs = [m for m in messages if m.role == Role.USER]
        code_feedback = [m for m in user_msgs if "Code Output:" in m.content]
        assert len(code_feedback) == 1


# ---------------------------------------------------------------------------
# Tests — State persistence (OperativeAgent pattern)
# ---------------------------------------------------------------------------


class TestMonitorStatePersistence:
    def test_recall_state(self):
        engine = _make_engine_no_tools()
        mem = MagicMock()
        mem.retrieve.return_value = "previous findings: error rate 5%"
        agent = MonitorAgent(
            engine, "test-model",
            operator_id="test-op",
            memory_backend=mem,
        )
        state = agent._recall_state()
        assert state == "previous findings: error rate 5%"
        mem.retrieve.assert_called_once_with("monitor:test-op:state")

    def test_recall_state_no_backend(self):
        engine = _make_engine_no_tools()
        agent = MonitorAgent(engine, "test-model", operator_id="test-op")
        assert agent._recall_state() == ""

    def test_recall_state_no_operator_id(self):
        engine = _make_engine_no_tools()
        mem = MagicMock()
        agent = MonitorAgent(engine, "test-model", memory_backend=mem)
        assert agent._recall_state() == ""

    def test_recall_state_exception(self):
        engine = _make_engine_no_tools()
        mem = MagicMock()
        mem.retrieve.side_effect = RuntimeError("db error")
        agent = MonitorAgent(
            engine, "test-model",
            operator_id="test-op",
            memory_backend=mem,
        )
        assert agent._recall_state() == ""

    def test_load_session(self):
        engine = _make_engine_no_tools()
        sess = MagicMock()
        mock_session = MagicMock()
        mock_session.messages = [
            {"role": "user", "content": "prev query"},
            {"role": "assistant", "content": "prev response"},
        ]
        sess.get_or_create.return_value = mock_session
        agent = MonitorAgent(
            engine, "test-model",
            operator_id="test-op",
            session_store=sess,
        )
        messages = agent._load_session()
        assert len(messages) == 2
        assert messages[0].role == Role.USER
        assert messages[0].content == "prev query"
        sess.get_or_create.assert_called_once_with("monitor:test-op")

    def test_load_session_no_store(self):
        engine = _make_engine_no_tools()
        agent = MonitorAgent(engine, "test-model", operator_id="test-op")
        assert agent._load_session() == []

    def test_save_session(self):
        engine = _make_engine_no_tools()
        sess = MagicMock()
        agent = MonitorAgent(
            engine, "test-model",
            operator_id="test-op",
            session_store=sess,
        )
        agent._save_session("hello", "world")
        assert sess.save_message.call_count == 2
        sess.save_message.assert_any_call(
            "monitor:test-op", {"role": "user", "content": "hello"},
        )
        sess.save_message.assert_any_call(
            "monitor:test-op", {"role": "assistant", "content": "world"},
        )

    def test_auto_persist_state(self):
        engine = _make_engine_no_tools()
        mem = MagicMock()
        agent = MonitorAgent(
            engine, "test-model",
            operator_id="test-op",
            memory_backend=mem,
        )
        agent._auto_persist_state("analysis complete")
        mem.store.assert_called_once_with(
            "monitor:test-op:state", "analysis complete",
        )

    def test_auto_persist_state_no_backend(self):
        engine = _make_engine_no_tools()
        agent = MonitorAgent(engine, "test-model", operator_id="test-op")
        # Should not raise
        agent._auto_persist_state("content")

    def test_auto_persist_truncates_long_content(self):
        engine = _make_engine_no_tools()
        mem = MagicMock()
        agent = MonitorAgent(
            engine, "test-model",
            operator_id="test-op",
            memory_backend=mem,
        )
        long_content = "x" * 2000
        agent._auto_persist_state(long_content)
        stored = mem.store.call_args[0][1]
        assert len(stored) == 1000

    def test_state_injected_into_system_prompt(self):
        engine = _make_engine_no_tools()
        mem = MagicMock()
        mem.retrieve.return_value = "previous state data"
        agent = MonitorAgent(
            engine, "test-model",
            operator_id="test-op",
            memory_backend=mem,
        )
        agent.run("Hello")
        messages = engine.generate.call_args[0][0]
        system_msg = messages[0].content
        assert "Previous State" in system_msg
        assert "previous state data" in system_msg

    def test_state_stored_by_tool_skips_auto_persist(self):
        """If agent explicitly stores state via memory_store, skip auto-persist."""
        engine = MagicMock()
        engine.engine_id = "mock"
        state_key = "monitor:test-op:state"
        engine.generate.side_effect = [
            {
                "content": "",
                "tool_calls": [{
                    "id": "c1", "name": "memory_store",
                    "arguments": json.dumps({"key": state_key, "value": "manual"}),
                }],
                "usage": {
                    "prompt_tokens": 5,
                    "completion_tokens": 3,
                    "total_tokens": 8,
                },
                "model": "test-model",
                "finish_reason": "tool_calls",
            },
            {
                "content": "Done.",
                "usage": {
                    "prompt_tokens": 15,
                    "completion_tokens": 3,
                    "total_tokens": 18,
                },
                "model": "test-model",
                "finish_reason": "stop",
            },
        ]

        class _MemoryStoreStub(BaseTool):
            tool_id = "memory_store"

            @property
            def spec(self) -> ToolSpec:
                return ToolSpec(
                    name="memory_store",
                    description="Store in memory.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "key": {"type": "string"},
                            "value": {"type": "string"},
                        },
                        "required": ["key", "value"],
                    },
                )

            def execute(self, **params) -> ToolResult:
                return ToolResult(
                    tool_name="memory_store",
                    content="stored",
                    success=True,
                )

        mem = MagicMock()
        mem.retrieve.return_value = ""
        agent = MonitorAgent(
            engine, "test-model",
            tools=[_MemoryStoreStub()],
            operator_id="test-op",
            memory_backend=mem,
        )
        agent.run("Store state")
        # auto_persist should NOT be called since tool stored state explicitly
        mem.store.assert_not_called()


# ---------------------------------------------------------------------------
# Tests — Context and events
# ---------------------------------------------------------------------------


class TestMonitorContextAndEvents:
    def test_with_context_conversation(self):
        engine = _make_engine_no_tools()
        agent = MonitorAgent(engine, "test-model")
        conv = Conversation()
        conv.add(Message(role=Role.SYSTEM, content="Be helpful."))
        ctx = AgentContext(conversation=conv)
        agent.run("Hi", context=ctx)
        messages = engine.generate.call_args[0][0]
        # System prompt + context system + user
        assert any(m.content == "Be helpful." for m in messages)

    def test_event_bus_agent_events(self):
        bus = EventBus(record_history=True)
        engine = _make_engine_no_tools()
        agent = MonitorAgent(engine, "test-model", bus=bus)
        agent.run("Hello")
        event_types = [e.event_type for e in bus.history]
        assert EventType.AGENT_TURN_START in event_types
        assert EventType.AGENT_TURN_END in event_types

    def test_event_bus_tool_events(self):
        bus = EventBus(record_history=True)
        engine = _make_engine_with_tool_call()
        agent = MonitorAgent(
            engine, "test-model", tools=[_CalculatorStub()], bus=bus,
        )
        agent.run("Calc 2+2")
        event_types = [e.event_type for e in bus.history]
        assert EventType.TOOL_CALL_START in event_types
        assert EventType.TOOL_CALL_END in event_types

    def test_custom_system_prompt_used(self):
        engine = _make_engine_no_tools()
        agent = MonitorAgent(
            engine, "test-model",
            system_prompt="Custom monitor prompt",
        )
        agent.run("Hello")
        messages = engine.generate.call_args[0][0]
        assert messages[0].content == "Custom monitor prompt"

    def test_default_system_prompt_contains_protocol(self):
        engine = _make_engine_no_tools()
        agent = MonitorAgent(engine, "test-model")
        agent.run("Hello")
        messages = engine.generate.call_args[0][0]
        system_msg = messages[0].content
        assert "Monitor Agent" in system_msg
        assert "TOOLS" in system_msg
        assert "CODE" in system_msg


# ---------------------------------------------------------------------------
# Tests — Continuation on length
# ---------------------------------------------------------------------------


class TestMonitorContinuation:
    def test_check_continuation_on_length(self):
        engine = MagicMock()
        engine.engine_id = "mock"
        engine.generate.side_effect = [
            {
                "content": "partial...",
                "usage": {
                    "prompt_tokens": 5,
                    "completion_tokens": 10,
                    "total_tokens": 15,
                },
                "model": "test-model",
                "finish_reason": "length",
            },
            {
                "content": " complete.",
                "usage": {
                    "prompt_tokens": 20,
                    "completion_tokens": 5,
                    "total_tokens": 25,
                },
                "model": "test-model",
                "finish_reason": "stop",
            },
        ]
        agent = MonitorAgent(engine, "test-model")
        result = agent.run("Hello")
        assert result.content == "partial... complete."


# ---------------------------------------------------------------------------
# Tests — Registry
# ---------------------------------------------------------------------------


class TestMonitorRegistry:
    def test_registered(self):
        from openjarvis.core.registry import AgentRegistry
        # Re-register after autouse fixture clears the registry
        if not AgentRegistry.contains("monitor"):
            AgentRegistry.register_value("monitor", MonitorAgent)
        assert AgentRegistry.contains("monitor")
        cls = AgentRegistry.get("monitor")
        assert cls is MonitorAgent


__all__ = [
    "TestMonitorInit",
    "TestMonitorToolCalling",
    "TestMonitorCodeExecution",
    "TestMonitorHybridLoop",
    "TestMonitorStatePersistence",
    "TestMonitorContextAndEvents",
    "TestMonitorContinuation",
    "TestMonitorRegistry",
]
