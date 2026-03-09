# Silent Exception Remediation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add `logger.warning()` to ~90-100 silent exception handlers across the codebase, making failures observable without changing runtime behavior.

**Architecture:** File-by-file remediation in priority order (critical path → user-facing → agents/tools → supporting systems). Each file gets `import logging` + `logger = logging.getLogger(__name__)` if missing, then each bare `except` block gets a contextual log message. Import guards (`except ImportError: pass`) are untouched.

**Tech Stack:** Python `logging` module (standard library only). No new dependencies.

---

## Universal Rules

1. **Add logger setup** if missing at top of file (after imports):
   ```python
   import logging

   logger = logging.getLogger(__name__)
   ```

2. **Transform patterns:**
   - `except Exception: pass` → `except Exception as exc: logger.warning("Context: %s", exc)`
   - `except X: return default` → `except X as exc: logger.warning("Context: %s", exc); return default`
   - `except X: continue` → `except X as exc: logger.warning("Context: %s", exc); continue`
   - Best-effort blocks (with existing comments) → `except X as exc: logger.debug("Context: %s", exc)`

3. **Do NOT touch** `except ImportError: pass` blocks.

4. **Use `%s` style** (not f-strings) in log messages per Python logging best practice.

5. **Log messages** should describe *what failed*, not the exception type. Good: `"Failed to inject memory context"`. Bad: `"Exception occurred"`.

---

### Task 1: system.py — Core composition layer

**Files:**
- Modify: `src/openjarvis/system.py`
- Test: existing tests via `uv run pytest tests/ -v --tb=short -x`

**Step 1: Add logging import**

Add after the existing imports (line 5 area):
```python
import logging

logger = logging.getLogger(__name__)
```

**Step 2: Remediate all silent blocks**

| Line | Current | New log message | Level |
|------|---------|----------------|-------|
| 82-83 | `except Exception: pass` | `"Failed to inject memory context: %s"` | warning |
| 259-260 | `except Exception: pass` (in `_build_tools`) | `"Failed to build tool %r: %s", name, exc` | warning |
| 474-475 | `except Exception: pass` (speech backend) | `"Failed to initialize speech backend: %s"` | warning |
| 527-528 | `except Exception: pass` (model resolution) | `"Failed to list models from engine: %s"` | warning |
| 558-559 | `except Exception: pass` (security guardrails) | `"Failed to set up security guardrails: %s"` | warning |
| 570-571 | `except Exception: return None` (telemetry store) | `"Failed to set up telemetry store: %s"` | warning |
| 582-583 | `except Exception: pass` + `return None` (memory) | `"Failed to resolve memory backend: %s"` | warning |
| 707-708 | `except Exception: return None` (channel) | `"Failed to resolve channel backend %r: %s", key, exc` | warning |
| 753-754 | `except Exception: pass` (external MCP) | `"Failed to discover external MCP tools: %s"` | warning |
| 755-756 | `except (json..., TypeError): pass` (MCP JSON) | `"Failed to parse MCP server config: %s"` | warning |
| 803-804 | `except Exception: return None` (sandbox) | `"Failed to set up container sandbox: %s"` | warning |
| 835-836 | `except Exception: return None, None` (scheduler) | `"Failed to set up task scheduler: %s"` | warning |
| 854-855 | `except Exception: return None` (workflow) | `"Failed to set up workflow engine: %s"` | warning |
| 873-874 | `except Exception: return None` (sessions) | `"Failed to set up session store: %s"` | warning |
| 887-888 | `except Exception: return None` (capabilities) | `"Failed to set up capability policy: %s"` | warning |
| 918-919 | `except Exception: return None` (learning) | `"Failed to set up learning orchestrator: %s"` | warning |

Example transformation for line 82:
```python
# Before:
            except Exception:
                pass

# After:
            except Exception as exc:
                logger.warning("Failed to inject memory context: %s", exc)
```

Example transformation for line 570 (return pattern):
```python
# Before:
        except Exception:
            return None

# After:
        except Exception as exc:
            logger.warning("Failed to set up telemetry store: %s", exc)
            return None
```

**Step 3: Run lint**

```bash
uv run ruff check src/openjarvis/system.py
```
Expected: PASS (zero warnings)

**Step 4: Run tests**

```bash
uv run pytest tests/test_system.py tests/test_system_builder.py -v --tb=short -x
```
Expected: All pass (no behavioral changes)

**Step 5: Commit**

```bash
git add src/openjarvis/system.py
git commit -m "fix: add logging to silent exception handlers in system.py"
```

---

### Task 2: sdk.py — Public API

**Files:**
- Modify: `src/openjarvis/sdk.py`

**Step 1: Add logging import**

Add after existing imports:
```python
import logging

logger = logging.getLogger(__name__)
```

**Step 2: Remediate all silent blocks**

| Line | Current | New log message | Level |
|------|---------|----------------|-------|
| 177-178 | `except Exception: pass` (telemetry init) | `"Failed to initialize telemetry store: %s"` | warning |
| 189-190 | `except Exception: pass` (audit logger) | `"Failed to initialize security audit logger: %s"` | warning |
| 245-246 | `except Exception: pass` (security guardrails, has comment) | `"Failed to set up security guardrails: %s"` | debug |
| 257-258 | `except Exception: pass` (energy monitor) | `"Failed to create energy monitor: %s"` | debug |
| 451-452 | `except Exception: pass` (model resolution) | `"Failed to list models from engine: %s"` | warning |
| 522-523 | `except Exception: pass` (agent context injection) | `"Failed to inject memory context for agent: %s"` | warning |
| 558-559 | `except Exception: pass` (_inject_context) | `"Failed to inject memory context: %s"` | warning |
| 579-580 | `except Exception: pass` (energy close) | `"Error closing energy monitor: %s"` | debug |
| 585-586 | `except Exception: pass` (telem close) | `"Error closing telemetry store: %s"` | debug |
| 591-592 | `except Exception: pass` (audit close) | `"Error closing audit logger: %s"` | debug |

Note: Close/cleanup blocks use `debug` level — errors during shutdown are low priority.

**Step 3: Run lint**

```bash
uv run ruff check src/openjarvis/sdk.py
```

**Step 4: Run tests**

```bash
uv run pytest tests/test_sdk.py -v --tb=short -x
```

**Step 5: Commit**

```bash
git add src/openjarvis/sdk.py
git commit -m "fix: add logging to silent exception handlers in sdk.py"
```

---

### Task 3: engine/ollama.py + engine/_openai_compat.py — Inference path

**Files:**
- Modify: `src/openjarvis/engine/ollama.py`
- Modify: `src/openjarvis/engine/_openai_compat.py`

**Step 1: Add logging import to both files**

```python
import logging

logger = logging.getLogger(__name__)
```

**Step 2: Remediate ollama.py**

| Line | Current | New log message | Level |
|------|---------|----------------|-------|
| 176-177 | `except (...): return []` (list_models) | `"Failed to list models from Ollama at %s: %s", self._host, exc` | warning |
| 185-186 | `except Exception: return False` (health) | `"Ollama health check failed at %s: %s", self._host, exc` | debug |

Note: `health()` uses `debug` since it's called frequently during discovery and a failure is expected when the engine is down.

**Step 3: Remediate _openai_compat.py**

| Line | Current | New log message | Level |
|------|---------|----------------|-------|
| 138-139 | `except (...): return []` (list_models) | `"Failed to list models from %s at %s: %s", self.engine_id, self._host, exc` | warning |
| 147-148 | `except Exception: return False` (health) | `"%s health check failed at %s: %s", self.engine_id, self._host, exc` | debug |

**Step 4: Run lint + tests**

```bash
uv run ruff check src/openjarvis/engine/ollama.py src/openjarvis/engine/_openai_compat.py
uv run pytest tests/engine/ -v --tb=short -x
```

**Step 5: Commit**

```bash
git add src/openjarvis/engine/ollama.py src/openjarvis/engine/_openai_compat.py
git commit -m "fix: add logging to silent exception handlers in engine backends"
```

---

### Task 4: cli/ask.py — Main CLI command

**Files:**
- Modify: `src/openjarvis/cli/ask.py`

**Step 1: Add logging import**

```python
import logging

logger = logging.getLogger(__name__)
```

**Step 2: Remediate all silent blocks**

| Line | Current | New log message | Level |
|------|---------|----------------|-------|
| 55-56 | `except Exception: return None` (_get_memory_backend) | `"Failed to initialize memory backend: %s"` | warning |
| 146-147 | `except Exception: pass` (agent context injection) | `"Failed to inject memory context for agent: %s"` | warning |
| 330-331 | `except Exception: pass` (telem store, has "best-effort" comment) | `"Failed to initialize telemetry store: %s"` | debug |
| 361-362 | `except Exception: pass` (energy monitor, has "best-effort" comment) | `"Failed to create energy monitor: %s"` | debug |
| 422-424 | `except Exception: pass` (telem close) | `"Error closing telemetry store: %s"` | debug |
| 450-451 | `except Exception: pass` (context injection, has "best-effort" comment) | `"Failed to inject memory context: %s"` | debug |
| 483-484 | `except Exception: pass` (energy close) | `"Error closing energy monitor: %s"` | debug |
| 488-489 | `except Exception: pass` (telem close) | `"Error closing telemetry store: %s"` | debug |

**Step 3: Run lint + tests**

```bash
uv run ruff check src/openjarvis/cli/ask.py
uv run pytest tests/cli/ -v --tb=short -x
```

**Step 4: Commit**

```bash
git add src/openjarvis/cli/ask.py
git commit -m "fix: add logging to silent exception handlers in cli/ask.py"
```

---

### Task 5: cli/quickstart_cmd.py + engine/_discovery.py

**Files:**
- Modify: `src/openjarvis/cli/quickstart_cmd.py`
- Modify: `src/openjarvis/engine/_discovery.py`

**Step 1: Add logging import to both files**

**Step 2: Remediate quickstart_cmd.py**

| Line | Current | New log message | Level |
|------|---------|----------------|-------|
| 30-31 | `except Exception: return False` | `"Engine health check failed for %r: %s", engine_key, exc` | warning |
| 46-47 | `except Exception: return False` | `"Model availability check failed for %r: %s", engine_key, exc` | warning |

**Step 3: Remediate _discovery.py**

| Line | Current | New log message | Level |
|------|---------|----------------|-------|
| 50-51 | `except Exception: continue` | `"Engine %r failed during discovery: %s", key, exc` | debug |
| 70-71 | `except Exception: result[key] = []` | `"Failed to list models for engine %r: %s", key, exc` | debug |
| 88-89 | `except Exception: pass` | `"Engine %r health check failed: %s", engine_key, exc` | debug |
| 99-100 | `except Exception: pass` | `"Default engine %r health check failed: %s", default_key, exc` | debug |

Note: Discovery blocks use `debug` — these fire frequently during startup probe and most "failures" are simply engines that aren't running.

**Step 4: Run lint + tests**

```bash
uv run ruff check src/openjarvis/cli/quickstart_cmd.py src/openjarvis/engine/_discovery.py
uv run pytest tests/engine/ tests/cli/ -v --tb=short -x
```

**Step 5: Commit**

```bash
git add src/openjarvis/cli/quickstart_cmd.py src/openjarvis/engine/_discovery.py
git commit -m "fix: add logging to silent exception handlers in quickstart + discovery"
```

---

### Task 6: agents/monitor_operative.py

**Files:**
- Modify: `src/openjarvis/agents/monitor_operative.py`

This file already has `logger = logging.getLogger(__name__)` and uses `logger.debug()` in many places. Only 3 blocks need attention.

**Step 1: Remediate remaining silent blocks**

| Line | Current | New log message | Level |
|------|---------|----------------|-------|
| 268-269 | `except (json.JSONDecodeError, TypeError): pass` | `"Failed to parse tool call arguments for state tracking: %s"` | debug |
| 443-444 | `except Exception: pass` (memory store in causality) | `"Failed to store causality relation in memory: %s"` | debug |
| 485-486 | `except Exception: pass` (memory store in structured) | `"Failed to store structured data in memory for tool %s: %s", tool_name, exc` | debug |

**Step 2: Run lint + tests**

```bash
uv run ruff check src/openjarvis/agents/monitor_operative.py
uv run pytest tests/agents/test_monitor_operative.py -v --tb=short -x
```

**Step 3: Commit**

```bash
git add src/openjarvis/agents/monitor_operative.py
git commit -m "fix: add logging to remaining silent exception handlers in monitor_operative"
```

---

### Task 7: security/ modules

**Files:**
- Modify: `src/openjarvis/security/signing.py`
- Modify: `src/openjarvis/security/capabilities.py`
- Modify: `src/openjarvis/security/subprocess_sandbox.py`

**Step 1: Add logging import to each file (if missing)**

**Step 2: Remediate signing.py**

| Line | Current | New log message | Level |
|------|---------|----------------|-------|
| 91-92 | `except Exception: return False` | `"Signature verification failed: %s"` | warning |
| 104-105 | `except Exception: pass` | `"Failed to generate key pair: %s"` | warning |

**Step 3: Remediate capabilities.py**

| Line | Current | New log message | Level |
|------|---------|----------------|-------|
| 141-142 | `except (json.JSONDecodeError, KeyError, TypeError): pass` | `"Failed to parse capability policy: %s"` | warning |

**Step 4: Remediate subprocess_sandbox.py**

| Line | Current | New log message | Level |
|------|---------|----------------|-------|
| 52-53 | `except (OSError, ProcessLookupError): pass` | `"Failed to terminate process %d: %s", pid, exc` | debug |
| 56-57 | `except (OSError, ProcessLookupError): pass` | `"Failed to kill process %d: %s", pid, exc` | debug |

Process cleanup blocks use `debug` — failure to kill an already-dead process is expected.

**Step 5: Run lint + tests**

```bash
uv run ruff check src/openjarvis/security/
uv run pytest tests/security/ -v --tb=short -x
```

**Step 6: Commit**

```bash
git add src/openjarvis/security/signing.py src/openjarvis/security/capabilities.py src/openjarvis/security/subprocess_sandbox.py
git commit -m "fix: add logging to silent exception handlers in security modules"
```

---

### Task 8: mcp/server.py

**Files:**
- Modify: `src/openjarvis/mcp/server.py`

**Step 1: Add logging import if missing**

**Step 2: Remediate silent blocks**

| Line | Current | New log message | Level |
|------|---------|----------------|-------|
| 144 | `except Exception: pass` (tool instantiation) | `"Failed to instantiate tool %r: %s", tool_cls, exc` | warning |
| 157 | `except Exception: pass` (user tool) | `"Failed to register user tool: %s"` | warning |
| 159 | `except Exception: pass` (outer catch) | `"Failed to discover tools from registry: %s"` | warning |

**Step 3: Run lint + tests**

```bash
uv run ruff check src/openjarvis/mcp/server.py
uv run pytest tests/mcp/ -v --tb=short -x
```

**Step 4: Commit**

```bash
git add src/openjarvis/mcp/server.py
git commit -m "fix: add logging to silent exception handlers in mcp/server.py"
```

---

### Task 9: telemetry/ modules

**Files:**
- Modify: `src/openjarvis/telemetry/gpu_monitor.py`
- Modify: `src/openjarvis/telemetry/energy_rapl.py`
- Modify: `src/openjarvis/telemetry/energy_apple.py`
- Modify: `src/openjarvis/telemetry/energy_amd.py`
- Modify: `src/openjarvis/telemetry/energy_nvidia.py`
- Modify: `src/openjarvis/telemetry/energy_monitor.py`
- Modify: `src/openjarvis/telemetry/session.py`
- Modify: `src/openjarvis/telemetry/store.py`
- Modify: `src/openjarvis/telemetry/vllm_metrics.py`
- Modify: `src/openjarvis/telemetry/aggregator.py`

All telemetry blocks use `logger.debug()` — telemetry is best-effort and failures during GPU/energy measurement are expected when hardware or drivers aren't present.

**Step 1: Add logging import to each file that lacks it**

**Step 2: Remediate gpu_monitor.py**

| Line | New log message | Level |
|------|----------------|-------|
| 132-133 | `"GPU monitor initialization failed: %s"` | debug |
| 144-145 | `"GPU monitor availability check failed: %s"` | debug |
| 169-170 | `"Failed to read GPU metrics: %s"` | debug |
| 309-310 | `"Failed to shut down GPU monitor: %s"` | debug |

**Step 3: Remediate energy_rapl.py**

| Line | New log message | Level |
|------|----------------|-------|
| 32-33 | `"Failed to read RAPL domain name: %s"` | debug |
| 39-40 | `"Failed to read RAPL energy: %s"` | debug |
| 47-48 | `"Failed to read RAPL max energy: %s"` | debug |
| 96-97 | `"RAPL monitor initialization failed: %s"` | debug |

**Step 4: Remediate energy_apple.py**

| Line | New log message | Level |
|------|----------------|-------|
| 53-54 | `"Failed to detect Apple Silicon chip brand: %s"` | debug |
| 81-82 | `"Failed to read Apple energy: %s"` | debug |

**Step 5: Remediate energy_amd.py**

| Line | New log message | Level |
|------|----------------|-------|
| 46-47 | `"AMD SMI initialization failed: %s"` | debug |
| 58-59 | `"AMD energy monitor availability check failed: %s"` | debug |
| 76-77 | `"Failed to read AMD GPU power: %s"` | debug |
| 126-127 | `"Failed to shut down AMD energy monitor: %s"` | debug |

**Step 6: Remediate energy_nvidia.py**

| Line | New log message | Level |
|------|----------------|-------|
| 59-60 | `"NVIDIA energy monitor initialization failed: %s"` | debug |
| 69-70 | `"NVIDIA energy monitor availability check failed: %s"` | debug |
| 81-82 | `"NVIDIA energy query support check failed: %s"` | debug |
| 97-98 | `"Failed to read NVIDIA GPU power: %s"` | debug |
| 119-120 | `"Failed to shut down NVIDIA energy monitor: %s"` | debug |
| 243-244 | `"Failed to shut down NVIDIA energy monitor: %s"` | debug |

**Step 7: Remediate energy_monitor.py**

| Line | New log message | Level |
|------|----------------|-------|
| 121-122, 128-129, 135-136, 142-143 | `"Failed to create %s energy monitor: %s", vendor, exc` | debug |
| 157-158 | `"Energy monitor candidate failed: %s"` | debug |

**Step 8: Remediate session.py, store.py, vllm_metrics.py, aggregator.py**

`session.py` line 110: `"Failed to record telemetry session: %s"` (debug)

`store.py` line 125-126: Keep as-is (sqlite column already exists is truly noise) — add comment only: `# Column already exists — safe to ignore`
`store.py` line 185-186: `"Failed to record telemetry event: %s"` (debug)

`vllm_metrics.py` lines 51-52, 57-58, 63-64, 116-117: `"Failed to parse vLLM metric line: %s"` (debug)
`vllm_metrics.py` line 132-133: `"Failed to fetch vLLM metrics from %s: %s"` (debug)

`aggregator.py` line 114-115: `"Telemetry aggregator table check failed: %s"` (debug)

**Step 9: Run lint + tests**

```bash
uv run ruff check src/openjarvis/telemetry/
uv run pytest tests/telemetry/ -v --tb=short -x
```

**Step 10: Commit**

```bash
git add src/openjarvis/telemetry/
git commit -m "fix: add debug logging to silent exception handlers in telemetry modules"
```

---

### Task 10: learning/ modules

**Files:**
- Modify: `src/openjarvis/learning/sft_policy.py`
- Modify: `src/openjarvis/learning/agent_advisor.py`
- Modify: `src/openjarvis/learning/icl_updater.py`
- Modify: `src/openjarvis/learning/router.py`
- Modify: `src/openjarvis/learning/training/lora.py`
- Modify: `src/openjarvis/learning/orchestrator/sft_trainer.py`
- Modify: `src/openjarvis/learning/orchestrator/policy_model.py`
- Modify: `src/openjarvis/learning/orchestrator/grpo_trainer.py`
- Modify: `src/openjarvis/learning/optimize/store.py`
- Modify: `src/openjarvis/learning/optimize/llm_optimizer.py`

**Step 1: Add logging import to each file that lacks it**

**Step 2: Remediate learning policy files**

`sft_policy.py` line 34-35: `"SFT policy update failed: %s"` (warning)
`agent_advisor.py` line 34-35: `"Agent advisor analysis failed: %s"` (warning)
`agent_advisor.py` line 60-61: `"Failed to parse agent advisor recommendation: %s"` (debug)
`icl_updater.py` line 43-44: `"ICL updater failed: %s"` (warning)
`router.py` line 46-47: `"Failed to compute model score: %s"` (debug)

**Step 3: Remediate training files**

`lora.py` line 362-363: Add comment `# fall through to manual format` and `logger.debug("Auto chat template failed, using manual format: %s", exc)` (debug)

`sft_trainer.py` line 175-176: Add `logger.debug("Auto chat template failed, using fallback: %s", exc)` (debug)

`policy_model.py` line 101-102: `logger.debug("FP8 not available, falling back to BF16: %s", exc)` (debug) — import guard, but for a runtime feature rather than a library

`grpo_trainer.py` line 163-164: `logger.debug("FP8 not available for GRPO: %s", exc)` (debug) — same pattern

**Step 4: Remediate optimize files**

`store.py` line 107-108: Keep as-is, add comment `# Column already exists — safe to ignore`
`store.py` lines 357-358, 369-370, 436-437, 450-451, 472-473: `"Failed to parse stored JSON: %s"` (debug) — these are JSON deserialization of stored data, benign when data is missing/corrupted

`llm_optimizer.py` lines 597-598, 610-611, 633-634, 645-646, 656-657: `"Failed to parse LLM optimizer JSON response: %s"` (debug) — LLM output parsing, failures are expected

**Step 5: Run lint + tests**

```bash
uv run ruff check src/openjarvis/learning/
uv run pytest tests/learning/ -v --tb=short -x
```

**Step 6: Commit**

```bash
git add src/openjarvis/learning/
git commit -m "fix: add logging to silent exception handlers in learning modules"
```

---

### Task 11: Remaining files (bench, a2a, core, litellm, sandbox)

**Files:**
- Modify: `src/openjarvis/bench/throughput.py`
- Modify: `src/openjarvis/bench/energy.py`
- Modify: `src/openjarvis/bench/latency.py`
- Modify: `src/openjarvis/a2a/tool.py`
- Modify: `src/openjarvis/core/events.py`
- Modify: `src/openjarvis/engine/litellm.py`
- Modify: `src/openjarvis/sandbox/mount_security.py` (if applicable)

**Step 1: Add logging import where missing**

**Step 2: Remediate bench files**

Warmup blocks are intentionally silent — add `logger.debug("Warmup request failed: %s", exc)` to each.
Measurement blocks already increment error counters — add `logger.debug("Measurement request failed: %s", exc)` alongside.

**Step 3: Remediate remaining files**

`a2a/tool.py` line 31: `logger.debug("Failed to fetch A2A agent description: %s", exc)` (debug)
`core/events.py` line 117: Add comment `# Callback already removed — idempotent`
`engine/litellm.py` line 94: `logger.debug("Failed to compute cost for LiteLLM call: %s", exc)` (debug)

**Step 4: Run lint + full test suite**

```bash
uv run ruff check src/ tests/
uv run pytest tests/ -v --tb=short -x -m "not live and not cloud and not slow"
```

**Step 5: Commit**

```bash
git add src/openjarvis/bench/ src/openjarvis/a2a/tool.py src/openjarvis/core/events.py src/openjarvis/engine/litellm.py
git commit -m "fix: add logging to silent exception handlers in remaining modules"
```

---

### Task 12: Final verification

**Step 1: Run full lint**

```bash
uv run ruff check src/ tests/
```
Expected: PASS (zero warnings)

**Step 2: Run full test suite**

```bash
uv run pytest tests/ -v --tb=short -m "not live and not cloud and not slow"
```
Expected: All tests pass

**Step 3: Verify no remaining bare silent blocks**

Search for remaining `except ... pass` blocks that lack both logging and comments (excluding import guards):
```bash
# This should return only import guards and intentionally commented blocks
grep -rn "except.*:$" src/openjarvis/ | grep -v "ImportError" | head -50
```

Review output to confirm all non-import-guard blocks now have either logging or a justifying comment.

**Step 4: Commit summary (if any stragglers found)**

```bash
git add -A
git commit -m "fix: address remaining silent exception handlers"
```
