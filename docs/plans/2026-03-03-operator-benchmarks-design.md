# Operator Benchmarks & MonitorOperativeAgent Design

**Date:** 2026-03-03
**Phase:** 25
**Status:** Approved

## Goal

Implement 5 long-horizon benchmarks for evaluating OpenJarvis Operator agents, and build a `MonitorOperativeAgent` that incorporates the best agentic strategies observed across these benchmarks.

## Benchmarks

### 1. LogHub (Easy)

- **Source:** [logpai/loghub](https://github.com/logpai/loghub) — HDFS_v1, BGL, Thunderbird
- **Task:** Classify log sessions/windows as anomaly or normal, with explanation
- **Scoring:** Binary exact match (anomaly/normal), optional LLM-judge for explanation quality
- **Category:** `agentic` | **Subjects:** `hdfs`, `bgl`, `thunderbird`
- **Environment:** None (pure text)
- **Agent strategy:** `observation_compression="window"`, `memory_extraction="scratchpad"`, `task_decomposition="monolithic"`

### 2. AMA-Bench (Medium)

- **Source:** [AMA-Bench GitHub](https://github.com/AMA-Bench/AMA-Bench) — 3,696 QA pairs across 308 trajectories
- **Task:** Answer questions about pre-recorded agent trajectories (recall, causal inference, state updating, state abstraction)
- **Scoring:** LLM-judge (Qwen3-32B protocol)
- **Category:** `agentic` | **Subjects:** `recall`, `causal_inference`, `state_updating`, `state_abstraction`
- **Environment:** None (trajectories are pre-recorded data)
- **Episode mode:** Group QA pairs by trajectory. Ingest trajectory into memory once, answer N questions sequentially
- **Agent strategy:** `memory_extraction="causality_graph"`, `retrieval="hybrid_with_self_eval"`, `observation_compression="summarize"`

### 3. LifelongAgentBench (Medium-Hard)

- **Source:** [arXiv:2505.11942](https://arxiv.org/abs/2505.11942) — Sequential tasks across DB, OS, KG environments
- **Task:** Complete tasks that depend on knowledge accumulated from previous tasks
- **Scoring:** Environment-state validation (did the DB/OS/KG end up in correct state?)
- **Category:** `agentic` | **Subjects:** `database`, `os`, `knowledge_graph`
- **Environment:** Docker containers (PostgreSQL, Ubuntu, in-memory KG) via `ContainerRunner`
- **Episode mode:** Critical. Tasks within an episode are sequentially dependent
- **Agent strategy:** `memory_extraction="causality_graph"`, `retrieval="hybrid_with_self_eval"`, `task_decomposition="phased"`

### 4. WebChoreArena (Hard)

- **Source:** [WebChoreArena GitHub](https://github.com/WebChoreArena/WebChoreArena) — 532 tasks on 4 Docker web environments
- **Task:** Complete tedious, memory-heavy web tasks (massive memory, calculation, long-term memory)
- **Scoring:** Three evaluator types: string_match, url_match, program_html
- **Category:** `agentic` | **Subjects:** `massive_memory`, `calculation`, `long_term_memory`
- **Environment:** 4 Docker containers (Magento shopping, Magento admin, Postmill forum, GitLab)
- **Agent strategy:** `observation_compression="axtree"`, `memory_extraction="causality_graph"`, `retrieval="hybrid"`, `task_decomposition="phased"`

### 5. WorkArena++ (Hard)

- **Source:** [ServiceNow/WorkArena](https://github.com/ServiceNow/WorkArena) — 682 enterprise workflow tasks
- **Task:** Complete multi-step enterprise workflows on live ServiceNow instance
- **Scoring:** Binary (all subtasks pass or fail). Environment-state validation via ServiceNow REST API
- **Category:** `agentic` | **Subjects:** `l2_planning`, `l2_retrieval`, `l2_decision`, `l2_memorization`, `l2_context`, `l3_*`
- **Environment:** ServiceNow developer instance (free tier)
- **Agent strategy:** `observation_compression="axtree"`, `memory_extraction="causality_graph"`, `retrieval="hybrid_with_self_eval"`, `task_decomposition="phased"`

## MonitorOperativeAgent

### Architecture

Extends `OperativeAgent` with 4 configurable strategies:

```
OperativeAgent (session + state persistence)
    ↓
MonitorOperativeAgent
    ├── MemoryExtractionStrategy  (what to store at each step)
    ├── ObservationStrategy       (how to compress observations)
    ├── RetrievalStrategy         (how to recall from memory)
    └── DecompositionStrategy     (how to break down tasks)
```

**Registry key:** `"monitor_operative"`

### Strategies

| Strategy | Options | Default |
|----------|---------|---------|
| Memory Extraction | `causality_graph`, `scratchpad`, `none` | `causality_graph` |
| Observation Compression | `summarize`, `axtree`, `window`, `none` | `summarize` |
| Retrieval | `hybrid_with_self_eval`, `hybrid`, `keyword` | `hybrid_with_self_eval` |
| Task Decomposition | `phased`, `monolithic` | `phased` |

### Memory Extraction: Causality Graph

After each tool result, an LLM call extracts:
- **Entities:** Named objects/states observed (e.g., "database table employees", "page /admin/products")
- **Relations:** Causal edges between states (e.g., "CREATE TABLE → table exists", "CLICK button → page changed")

Stored via `KnowledgeGraphMemory.add_entity()` / `add_relation()`. Text summary also stored via `HybridMemory.store()`.

### Retrieval: Hybrid with Self-Evaluation

1. Query both KG (graph traversal from relevant entities) and HybridMemory (BM25+dense)
2. Merge results, deduplicate
3. Ask LLM: "Given these retrieved facts, do I have enough context to answer/act?"
4. If insufficient, broaden search (increase top_k, expand graph traversal depth)

### Tool Set

```python
tools = [
    "think", "calculator", "code_interpreter",
    "memory_store", "memory_search", "memory_retrieve",
    "kg_add_entity", "kg_add_relation", "kg_query", "kg_neighbors",
    "browser_navigate", "browser_click", "browser_type",
    "browser_extract", "browser_screenshot", "browser_axtree",
    "file_read", "file_write", "web_search", "http_request",
    "shell_exec", "db_query",
]
```

### Operator Recipe

```toml
[operator]
name = "monitor"
description = "Long-horizon monitoring agent with structured memory"

[operator.agent]
type = "monitor_operative"
max_turns = 30
temperature = 0.2
tools = ["think", "calculator", "code_interpreter",
         "memory_store", "memory_search", "memory_retrieve",
         "kg_add_entity", "kg_add_relation", "kg_query", "kg_neighbors",
         "browser_navigate", "browser_click", "browser_type",
         "browser_extract", "browser_screenshot", "browser_axtree",
         "file_read", "file_write", "web_search", "http_request",
         "shell_exec", "db_query"]

[operator.strategies]
memory_extraction = "causality_graph"
observation_compression = "summarize"
retrieval = "hybrid_with_self_eval"
task_decomposition = "phased"
```

### Learning

- **Router:** `BanditRouterPolicy` (Thompson Sampling) — learn which strategy config works best per benchmark category
- **Agent:** `AgentConfigEvolver` — tune max_turns, temperature, tool set per benchmark from traces
- **Traces:** Full `TraceCollector` → `TrainingDataMiner` for SFT pair extraction

## Eval Framework Changes

### Episode Mode in EvalRunner

New `episode_mode` flag. When enabled:
- `DatasetProvider.iter_episodes()` yields `List[EvalRecord]` groups
- Records within an episode processed sequentially with shared agent state
- Agent state (memory, session) persists within episode, reset between episodes
- Episodes parallelizable across workers

### EnvironmentProvider ABC

```python
class EnvironmentProvider(ABC):
    def setup(self) -> Dict[str, Any]:
        """Start environment, return connection info"""
    def reset(self) -> None:
        """Reset state between tasks"""
    def validate(self, record: EvalRecord) -> Tuple[bool, Dict]:
        """Check environment state against expected outcome"""
    def teardown(self) -> None:
        """Stop environment"""
```

Implementations: `LifelongEnvironmentProvider`, `WebArenaEnvironmentProvider`, `WorkArenaEnvironmentProvider`

### New Tool: browser_axtree

Extracts accessibility tree from current Playwright page. Returns structured text representation of the DOM with element IDs, roles, names, and states. Used by top-performing agents on WebArena-family benchmarks.

## LM + Engine Configuration

- **Model:** Qwen3-32B (strong agentic reasoning, 32K context, good tool-calling)
- **Engine:** Ollama (default engine, straightforward setup)
- **Hardware:** GPU with >=24GB VRAM recommended for Qwen3-32B

## Testing

### Unit Tests

```
tests/evals/
├── test_loghub.py
├── test_ama_bench.py
├── test_lifelong_agent.py
├── test_webchorearena.py
├── test_workarena.py
├── test_episode_mode.py
├── test_environment_provider.py
tests/agents/
├── test_monitor_operative.py
├── test_observation_strategy.py
├── test_retrieval_strategy.py
tests/tools/
├── test_browser_axtree.py
```

### Sanity Check

5 queries from each benchmark with Qwen3-32B via Ollama:

```bash
uv run jarvis eval run -b loghub -m "qwen3:32b" -n 5 --backend jarvis-agent -v
uv run jarvis eval run -b ama-bench -m "qwen3:32b" -n 5 --backend jarvis-agent -v
uv run jarvis eval run -b lifelong-agent -m "qwen3:32b" -n 5 --backend jarvis-agent -v
uv run jarvis eval run -b webchorearena -m "qwen3:32b" -n 5 --backend jarvis-agent -v
uv run jarvis eval run -b workarena -m "qwen3:32b" -n 5 --backend jarvis-agent -v
```

## Implementation Order

1. **LogHub** — zero framework changes, pure data + scorer
2. **EvalRunner episode mode** — adds `iter_episodes()`, sequential processing
3. **EnvironmentProvider ABC** — base class for environment management
4. **MonitorOperativeAgent** — agent with 4 strategies
5. **AMA-Bench** — uses episode mode + causality graph extraction
6. **LifelongAgentBench** — uses episode mode + environment provider + Docker
7. **browser_axtree tool** — new Playwright accessibility tree tool
8. **WebChoreArena** — uses environment provider + browser tools
9. **WorkArena++** — uses environment provider + ServiceNow integration
10. **Unit tests** — comprehensive tests for all components
11. **Documentation** — CLAUDE.md, API surface updates
12. **Sanity check** — 5 queries per benchmark with Qwen3-32B

## Research Sources

- LogHub: [arXiv:2008.06448](https://arxiv.org/abs/2008.06448), [GitHub](https://github.com/logpai/loghub)
- AMA-Bench: [arXiv:2602.22769](https://arxiv.org/abs/2602.22769), [GitHub](https://github.com/AMA-Bench/AMA-Bench)
- LifelongAgentBench: [arXiv:2505.11942](https://arxiv.org/abs/2505.11942)
- WebChoreArena: [arXiv:2506.01952](https://arxiv.org/abs/2506.01952), [GitHub](https://github.com/WebChoreArena/WebChoreArena)
- WorkArena++: [NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/0b82662b6c32e887bb252a74d8cb2d5e-Paper-Datasets_and_Benchmarks_Track.pdf), [GitHub](https://github.com/ServiceNow/WorkArena)
