"""MonitorAgent — hybrid agent combining tool calling with code execution.

Combines three patterns from existing agents:
1. OrchestratorAgent's function-calling tool loop
2. RLMAgent's Python code block extraction and execution
3. OperativeAgent's cross-session state persistence

The hybrid loop allows the agent to freely interleave tool calls (for memory,
KG, grep, files, web) with inline Python code execution (for data processing,
calculation, log parsing) within a single session.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, List, Optional

from openjarvis.agents._stubs import AgentContext, AgentResult, ToolUsingAgent
from openjarvis.core.events import EventBus
from openjarvis.core.registry import AgentRegistry
from openjarvis.core.types import Message, Role, ToolCall, ToolResult
from openjarvis.engine._stubs import InferenceEngine
from openjarvis.tools._stubs import BaseTool

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

MONITOR_SYSTEM_PROMPT = """\
You are a Monitor Agent with two capabilities:

1. TOOLS: Call any available tool via function calling (memory, KG, grep, web)
2. CODE: Write Python in ```python blocks for data processing, calculation, and analysis

## Protocol
- For searching/filtering: use shell_exec with grep/awk/sed
- For storing findings: use memory_store and kg_add_entity/kg_add_relation
- For retrieving context: use memory_search and kg_query/kg_neighbors
- For calculation: ALWAYS write Python code, never calculate in your head
- For log parsing: write Python with regex, or use shell_exec with grep
- For SQL queries: use database_query with the appropriate db_path
- For reasoning: use think to organize your thoughts

## State Management
- Your previous findings are available via memory_search
- Your causal model is in the knowledge graph, query with kg_neighbors
- Always store important findings with memory_store before finishing
- Always record causal patterns with kg_add_entity + kg_add_relation

## Analysis Strategy
1. Start by searching memory for prior knowledge about this topic
2. Query KG for related causal chains
3. Gather new evidence (read files, grep logs, query databases, fetch URLs)
4. Process evidence with Python code for precision
5. Update KG with new causal relationships
6. Store summary findings in memory
7. Synthesize and report

{tool_descriptions}"""


@AgentRegistry.register("monitor")
class MonitorAgent(ToolUsingAgent):
    """Hybrid agent combining tool calling with inline code execution.

    The hybrid loop processes each turn by checking the engine response for:
    a. ``tool_calls`` — executed via ToolExecutor (OrchestratorAgent pattern)
    b. ```python code blocks — executed via CodeInterpreterTool (RLM pattern)
    c. Neither — returned as the final answer

    Optionally supports cross-session state persistence via ``operator_id``,
    ``session_store``, and ``memory_backend`` (OperativeAgent pattern).
    """

    agent_id = "monitor"
    accepts_tools = True

    def __init__(
        self,
        engine: InferenceEngine,
        model: str,
        *,
        tools: Optional[List[BaseTool]] = None,
        bus: Optional[EventBus] = None,
        max_turns: int = 25,
        temperature: float = 0.3,
        max_tokens: int = 4096,
        system_prompt: Optional[str] = None,
        # OperativeAgent state management
        operator_id: Optional[str] = None,
        session_store: Optional[Any] = None,
        memory_backend: Optional[Any] = None,
        # RLM code execution
        enable_code_execution: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            engine, model, tools=tools, bus=bus,
            max_turns=max_turns, temperature=temperature,
            max_tokens=max_tokens,
        )
        self._system_prompt = system_prompt
        self._operator_id = operator_id
        self._session_store = session_store
        self._memory_backend = memory_backend
        self._enable_code_execution = enable_code_execution

    # ------------------------------------------------------------------
    # Main hybrid loop
    # ------------------------------------------------------------------

    def run(
        self,
        input: str,
        context: Optional[AgentContext] = None,
        **kwargs: Any,
    ) -> AgentResult:
        self._emit_turn_start(input)

        # Build system prompt with state context
        sys_parts: list[str] = []
        if self._system_prompt:
            sys_parts.append(self._system_prompt)
        else:
            tool_desc = self._build_tool_descriptions()
            try:
                sys_parts.append(
                    MONITOR_SYSTEM_PROMPT.format(tool_descriptions=tool_desc),
                )
            except KeyError:
                sys_parts.append(MONITOR_SYSTEM_PROMPT)

        # State recall from memory backend
        previous_state = self._recall_state()
        if previous_state:
            sys_parts.append(f"\n## Previous State\n{previous_state}")

        system_prompt = "\n\n".join(sys_parts) if sys_parts else None

        # Load session history
        session_messages = self._load_session()

        # Build messages
        messages = self._build_monitor_messages(
            input, context,
            system_prompt=system_prompt,
            session_messages=session_messages,
        )

        # Get OpenAI-format tool definitions
        openai_tools = self._executor.get_openai_tools() if self._tools else []

        all_tool_results: list[ToolResult] = []
        turns = 0
        content = ""
        state_stored_by_tool = False

        for _turn in range(self._max_turns):
            turns += 1

            gen_kwargs: dict[str, Any] = {}
            if openai_tools:
                gen_kwargs["tools"] = openai_tools

            result = self._generate(messages, **gen_kwargs)
            content = result.get("content", "")
            raw_tool_calls = result.get("tool_calls", [])

            # --- Branch A: tool calls ---
            if raw_tool_calls:
                tool_calls = [
                    ToolCall(
                        id=tc.get("id", f"call_{i}"),
                        name=tc.get("name", ""),
                        arguments=tc.get("arguments", "{}"),
                    )
                    for i, tc in enumerate(raw_tool_calls)
                ]

                messages.append(Message(
                    role=Role.ASSISTANT,
                    content=content,
                    tool_calls=tool_calls,
                ))

                for tc in tool_calls:
                    # Loop guard check
                    if self._loop_guard:
                        verdict = self._loop_guard.check_call(
                            tc.name, tc.arguments,
                        )
                        if verdict.blocked:
                            tool_result = ToolResult(
                                tool_name=tc.name,
                                content=f"Loop guard: {verdict.reason}",
                                success=False,
                            )
                            all_tool_results.append(tool_result)
                            messages.append(Message(
                                role=Role.TOOL,
                                content=tool_result.content,
                                tool_call_id=tc.id,
                                name=tc.name,
                            ))
                            continue

                    tool_result = self._executor.execute(tc)
                    all_tool_results.append(tool_result)

                    # Track explicit state storage
                    if tc.name == "memory_store" and self._operator_id:
                        try:
                            args = json.loads(tc.arguments)
                            state_key = f"monitor:{self._operator_id}:state"
                            if args.get("key", "") == state_key:
                                state_stored_by_tool = True
                        except (json.JSONDecodeError, TypeError):
                            pass

                    messages.append(Message(
                        role=Role.TOOL,
                        content=tool_result.content,
                        tool_call_id=tc.id,
                        name=tc.name,
                    ))
                continue

            # --- Branch B: Python code blocks ---
            code = self._extract_code(content) if self._enable_code_execution else None
            if code is not None:
                messages.append(Message(role=Role.ASSISTANT, content=content))

                code_output = self._execute_code(code)
                code_result = ToolResult(
                    tool_name="code_interpreter",
                    content=code_output,
                    success=not code_output.startswith("Blocked:")
                    and not code_output.startswith("Execution"),
                )
                all_tool_results.append(code_result)

                messages.append(Message(
                    role=Role.USER,
                    content=f"Code Output:\n{code_output}",
                ))
                continue

            # --- Branch C: final answer ---
            content = self._check_continuation(result, messages)
            break
        else:
            # Max turns exceeded
            self._save_session(input, content)
            return self._max_turns_result(
                all_tool_results, turns, content=content,
            )

        # Save session and auto-persist state
        self._save_session(input, content)
        if not state_stored_by_tool:
            self._auto_persist_state(content)

        self._emit_turn_end(turns=turns, content_length=len(content))
        return AgentResult(
            content=content,
            tool_results=all_tool_results,
            turns=turns,
        )

    # ------------------------------------------------------------------
    # Message building
    # ------------------------------------------------------------------

    def _build_monitor_messages(
        self,
        input: str,
        context: Optional[AgentContext],
        *,
        system_prompt: Optional[str] = None,
        session_messages: Optional[list[Message]] = None,
    ) -> list[Message]:
        """Build message list with system prompt, session history, and input."""
        messages: list[Message] = []
        if system_prompt:
            messages.append(Message(role=Role.SYSTEM, content=system_prompt))
        if session_messages:
            messages.extend(session_messages)
        if context and context.conversation.messages:
            messages.extend(context.conversation.messages)
        messages.append(Message(role=Role.USER, content=input))
        return messages

    def _build_tool_descriptions(self) -> str:
        """Build a text description of available tools for the system prompt."""
        if not self._tools:
            return ""
        from openjarvis.tools._stubs import build_tool_descriptions
        return build_tool_descriptions(self._tools)

    # ------------------------------------------------------------------
    # Code execution (RLM pattern)
    # ------------------------------------------------------------------

    def _execute_code(self, code: str) -> str:
        """Execute Python code via CodeInterpreterTool.

        Falls back to a direct subprocess call if the code_interpreter tool
        is not available in the tool set.
        """
        # Try to find code_interpreter in our tools
        for tool in self._tools:
            if getattr(tool, "tool_id", "") == "code_interpreter":
                result = tool.execute(code=code)
                return result.content

        # Fallback: use CodeInterpreterTool directly
        try:
            from openjarvis.tools.code_interpreter import CodeInterpreterTool
            interp = CodeInterpreterTool()
            result = interp.execute(code=code)
            return result.content
        except ImportError:
            return "Error: code_interpreter not available"

    @staticmethod
    def _extract_code(text: str) -> Optional[str]:
        """Extract the first ```python code block from text.

        Also matches bare ``` blocks. Returns None if no code block found.
        """
        m = re.search(r"```python\s*\n(.*?)```", text, re.DOTALL)
        if m:
            return m.group(1).strip()
        m = re.search(r"```\s*\n(.*?)```", text, re.DOTALL)
        if m:
            return m.group(1).strip()
        return None

    # ------------------------------------------------------------------
    # State persistence (OperativeAgent pattern)
    # ------------------------------------------------------------------

    def _recall_state(self) -> str:
        """Retrieve previous state from memory backend."""
        if not self._memory_backend or not self._operator_id:
            return ""
        state_key = f"monitor:{self._operator_id}:state"
        try:
            result = self._memory_backend.retrieve(state_key)
            if result:
                return result if isinstance(result, str) else str(result)
        except Exception:
            logger.debug(
                "No previous state for monitor %s", self._operator_id,
            )
        return ""

    def _load_session(self) -> list[Message]:
        """Load recent session history for this operator."""
        if not self._session_store or not self._operator_id:
            return []
        session_id = f"monitor:{self._operator_id}"
        try:
            session = self._session_store.get_or_create(session_id)
            if hasattr(session, "messages") and session.messages:
                recent = session.messages[-10:]
                return [
                    Message(
                        role=Role(m.get("role", "user")),
                        content=m.get("content", ""),
                    )
                    for m in recent
                    if isinstance(m, dict)
                ]
        except Exception:
            logger.debug(
                "Could not load session for monitor %s", self._operator_id,
            )
        return []

    def _save_session(self, input_text: str, response: str) -> None:
        """Save the tick's prompt and response to the session store."""
        if not self._session_store or not self._operator_id:
            return
        session_id = f"monitor:{self._operator_id}"
        try:
            self._session_store.save_message(
                session_id, {"role": "user", "content": input_text},
            )
            self._session_store.save_message(
                session_id, {"role": "assistant", "content": response},
            )
        except Exception:
            logger.debug(
                "Could not save session for monitor %s", self._operator_id,
            )

    def _auto_persist_state(self, content: str) -> None:
        """Auto-persist a state summary if agent didn't store explicitly."""
        if not self._memory_backend or not self._operator_id:
            return
        state_key = f"monitor:{self._operator_id}:state"
        try:
            summary = content[:1000] if content else ""
            self._memory_backend.store(state_key, summary)
        except Exception:
            logger.debug(
                "Could not auto-persist state for monitor %s",
                self._operator_id,
            )


__all__ = ["MonitorAgent", "MONITOR_SYSTEM_PROMPT"]
