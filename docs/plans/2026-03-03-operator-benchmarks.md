# Operator Benchmarks & MonitorOperativeAgent Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement 5 long-horizon benchmarks (LogHub, AMA-Bench, LifelongAgentBench, WebChoreArena, WorkArena++) and a `MonitorOperativeAgent` with structured memory extraction, observation compression, hybrid retrieval, and task decomposition strategies.

**Architecture:** Extend the eval framework with episode mode (sequential processing with shared agent state) and an `EnvironmentProvider` ABC for Docker/ServiceNow environments. Build `MonitorOperativeAgent` extending `OperativeAgent` with 4 configurable strategies. Each benchmark is a `DatasetProvider` + `Scorer` pair following existing patterns.

**Tech Stack:** Python 3.10+, HuggingFace `datasets`, Playwright (browser tools), Docker (environment management), SQLite (KnowledgeGraphMemory), existing OpenJarvis registries and ABCs.

---

## Task 1: LogHub Dataset Provider

**Files:**
- Create: `src/openjarvis/evals/datasets/loghub.py`
- Test: `tests/evals/test_loghub.py`

**Step 1: Write the failing test**

Create `tests/evals/test_loghub.py`:

```python
"""Tests for LogHub dataset provider."""

from openjarvis.evals.datasets.loghub import LogHubDataset


class TestLogHubDataset:
    def test_instantiation(self) -> None:
        ds = LogHubDataset()
        assert ds.dataset_id == "loghub"
        assert ds.dataset_name == "LogHub"

    def test_has_required_methods(self) -> None:
        ds = LogHubDataset()
        assert hasattr(ds, "load")
        assert hasattr(ds, "iter_records")
        assert hasattr(ds, "size")
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/evals/test_loghub.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write minimal implementation**

Create `src/openjarvis/evals/datasets/loghub.py`:

```python
"""LogHub log anomaly detection dataset.

Supports HDFS, BGL, and Thunderbird log datasets from
https://github.com/logpai/loghub for evaluating log analysis agents.
"""

from __future__ import annotations

import csv
import logging
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from openjarvis.evals.core.dataset import DatasetProvider
from openjarvis.evals.core.types import EvalRecord

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a log analysis expert. Analyze the following log session "
    "and determine if it indicates an anomaly or is normal behavior.\n\n"
    "Respond with exactly one of: ANOMALY or NORMAL\n"
    "Then provide a brief explanation of your reasoning."
)

_DATASETS = {
    "hdfs": {
        "hf_path": "logpai/loghub-HDFS-v1",
        "log_file": "HDFS.log",
        "label_file": "anomaly_label.csv",
        "mode": "session",  # group by block_id
    },
    "bgl": {
        "hf_path": "logpai/loghub-BGL",
        "log_file": "BGL.log",
        "mode": "window",  # fixed-size windows
        "window_size": 100,
    },
    "thunderbird": {
        "hf_path": "logpai/loghub-Thunderbird",
        "log_file": "Thunderbird.log",
        "mode": "window",
        "window_size": 100,
    },
}


class LogHubDataset(DatasetProvider):
    """LogHub log anomaly detection benchmark."""

    dataset_id = "loghub"
    dataset_name = "LogHub"

    def __init__(
        self,
        subset: str = "hdfs",
        cache_dir: Optional[str] = None,
    ) -> None:
        if subset not in _DATASETS:
            raise ValueError(
                f"Unknown LogHub subset: {subset}. "
                f"Choose from: {list(_DATASETS.keys())}"
            )
        self._subset = subset
        self._cache_dir = (
            Path(cache_dir) if cache_dir
            else Path.home() / ".cache" / "loghub"
        )
        self._records: List[EvalRecord] = []

    def load(
        self,
        *,
        max_samples: Optional[int] = None,
        split: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> None:
        meta = _DATASETS[self._subset]
        data_dir = self._cache_dir / self._subset

        if not data_dir.exists():
            self._download(meta, data_dir)

        if meta["mode"] == "session":
            records = self._load_session_mode(data_dir, meta)
        else:
            records = self._load_window_mode(data_dir, meta)

        if seed is not None:
            random.Random(seed).shuffle(records)
        if max_samples is not None:
            records = records[:max_samples]

        self._records = records

    def iter_records(self) -> Iterable[EvalRecord]:
        return iter(self._records)

    def size(self) -> int:
        return len(self._records)

    def _download(self, meta: Dict[str, Any], data_dir: Path) -> None:
        """Download dataset from HuggingFace."""
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            raise ImportError(
                "huggingface_hub required for LogHub download. "
                "Install with: pip install huggingface_hub"
            )
        data_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=meta["hf_path"],
            repo_type="dataset",
            local_dir=str(data_dir),
        )

    def _load_session_mode(
        self, data_dir: Path, meta: Dict[str, Any],
    ) -> List[EvalRecord]:
        """Load HDFS-style session-based records (group by block_id)."""
        log_path = data_dir / meta["log_file"]
        label_path = data_dir / meta["label_file"]

        # Load labels: block_id -> "Anomaly" / "Normal"
        labels: Dict[str, str] = {}
        if label_path.exists():
            with open(label_path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    bid = row.get("BlockId", "")
                    lbl = row.get("Label", "Normal")
                    labels[bid] = lbl

        # Group log lines by block_id
        import re
        block_pattern = re.compile(r"blk_[-]?\d+")
        sessions: Dict[str, List[str]] = {}

        with open(log_path, errors="replace") as f:
            for line in f:
                match = block_pattern.search(line)
                if match:
                    bid = match.group()
                    sessions.setdefault(bid, []).append(line.rstrip())

        records: List[EvalRecord] = []
        for bid, lines in sessions.items():
            label = labels.get(bid, "Normal")
            reference = "anomaly" if label == "Anomaly" else "normal"
            log_text = "\n".join(lines[:200])  # Cap at 200 lines per session

            problem = (
                f"{_SYSTEM_PROMPT}\n\n"
                f"Log session for block {bid} "
                f"({len(lines)} lines):\n```\n{log_text}\n```"
            )

            records.append(EvalRecord(
                record_id=f"loghub-{self._subset}-{bid}",
                problem=problem,
                reference=reference,
                category="agentic",
                subject=self._subset,
                metadata={
                    "block_id": bid,
                    "num_lines": len(lines),
                    "dataset": self._subset,
                    "label": label,
                },
            ))

        return records

    def _load_window_mode(
        self, data_dir: Path, meta: Dict[str, Any],
    ) -> List[EvalRecord]:
        """Load BGL/Thunderbird-style windowed records."""
        log_path = data_dir / meta["log_file"]
        window_size = meta.get("window_size", 100)

        records: List[EvalRecord] = []
        window: List[str] = []
        has_anomaly = False
        window_idx = 0

        with open(log_path, errors="replace") as f:
            for line in f:
                stripped = line.rstrip()
                # BGL/Thunderbird: first token is "-" (normal) or fault category
                is_anomalous = not stripped.startswith("-")
                if is_anomalous:
                    has_anomaly = True
                window.append(stripped)

                if len(window) >= window_size:
                    reference = "anomaly" if has_anomaly else "normal"
                    log_text = "\n".join(window)

                    problem = (
                        f"{_SYSTEM_PROMPT}\n\n"
                        f"Log window {window_idx} "
                        f"({len(window)} lines):\n```\n{log_text}\n```"
                    )

                    records.append(EvalRecord(
                        record_id=f"loghub-{self._subset}-w{window_idx}",
                        problem=problem,
                        reference=reference,
                        category="agentic",
                        subject=self._subset,
                        metadata={
                            "window_idx": window_idx,
                            "num_lines": len(window),
                            "dataset": self._subset,
                            "has_anomaly": has_anomaly,
                        },
                    ))

                    window = []
                    has_anomaly = False
                    window_idx += 1

        return records
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/evals/test_loghub.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/openjarvis/evals/datasets/loghub.py tests/evals/test_loghub.py
git commit -m "feat(evals): add LogHub dataset provider for log anomaly detection"
```

---

## Task 2: LogHub Scorer

**Files:**
- Create: `src/openjarvis/evals/scorers/loghub_scorer.py`
- Modify: `tests/evals/test_loghub.py`

**Step 1: Write the failing test**

Append to `tests/evals/test_loghub.py`:

```python
from unittest.mock import MagicMock
from openjarvis.evals.core.types import EvalRecord
from openjarvis.evals.scorers.loghub_scorer import LogHubScorer


def _mock_backend() -> MagicMock:
    backend = MagicMock()
    backend.generate.return_value = "A"
    return backend


class TestLogHubScorer:
    def test_instantiation(self) -> None:
        s = LogHubScorer(_mock_backend(), "test-model")
        assert s.scorer_id == "loghub"

    def test_exact_match_anomaly(self) -> None:
        s = LogHubScorer(_mock_backend(), "test-model")
        record = EvalRecord(
            record_id="test-1", problem="analyze logs",
            reference="anomaly", category="agentic",
        )
        is_correct, meta = s.score(record, "ANOMALY\nThe logs show errors.")
        assert is_correct is True
        assert meta["match_type"] == "exact"

    def test_exact_match_normal(self) -> None:
        s = LogHubScorer(_mock_backend(), "test-model")
        record = EvalRecord(
            record_id="test-2", problem="analyze logs",
            reference="normal", category="agentic",
        )
        is_correct, meta = s.score(record, "NORMAL - no issues detected")
        assert is_correct is True

    def test_empty_response(self) -> None:
        s = LogHubScorer(_mock_backend(), "test-model")
        record = EvalRecord(
            record_id="test-3", problem="analyze logs",
            reference="anomaly", category="agentic",
        )
        is_correct, meta = s.score(record, "")
        assert is_correct is False
        assert meta["reason"] == "empty_response"

    def test_wrong_classification(self) -> None:
        s = LogHubScorer(_mock_backend(), "test-model")
        record = EvalRecord(
            record_id="test-4", problem="analyze logs",
            reference="anomaly", category="agentic",
        )
        is_correct, meta = s.score(record, "NORMAL - everything looks fine")
        assert is_correct is False
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/evals/test_loghub.py::TestLogHubScorer -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write minimal implementation**

Create `src/openjarvis/evals/scorers/loghub_scorer.py`:

```python
"""Scorer for LogHub log anomaly detection benchmark."""

from __future__ import annotations

import re
from typing import Any, Dict, Optional, Tuple

from openjarvis.evals.core.scorer import LLMJudgeScorer
from openjarvis.evals.core.types import EvalRecord

_ANOMALY_PATTERN = re.compile(r"\bANOMAL(?:Y|OUS)\b", re.IGNORECASE)
_NORMAL_PATTERN = re.compile(r"\bNORMAL\b", re.IGNORECASE)


class LogHubScorer(LLMJudgeScorer):
    """Score log anomaly detection: extract ANOMALY/NORMAL classification."""

    scorer_id = "loghub"

    def score(
        self, record: EvalRecord, model_answer: str,
    ) -> Tuple[Optional[bool], Dict[str, Any]]:
        if not model_answer or not model_answer.strip():
            return False, {"reason": "empty_response"}

        reference = record.reference.lower().strip()

        # Extract classification from response
        has_anomaly = bool(_ANOMALY_PATTERN.search(model_answer))
        has_normal = bool(_NORMAL_PATTERN.search(model_answer))

        if has_anomaly and not has_normal:
            predicted = "anomaly"
        elif has_normal and not has_anomaly:
            predicted = "normal"
        elif has_anomaly and has_normal:
            # Ambiguous — check which appears first
            a_pos = _ANOMALY_PATTERN.search(model_answer).start()
            n_pos = _NORMAL_PATTERN.search(model_answer).start()
            predicted = "anomaly" if a_pos < n_pos else "normal"
        else:
            # Neither keyword found — use LLM judge fallback
            return self._llm_fallback(record, model_answer)

        is_correct = predicted == reference
        return is_correct, {
            "match_type": "exact",
            "predicted": predicted,
            "reference": reference,
        }

    def _llm_fallback(
        self, record: EvalRecord, model_answer: str,
    ) -> Tuple[Optional[bool], Dict[str, Any]]:
        """Use LLM judge when keyword extraction fails."""
        prompt = (
            f"A log analysis agent was asked to classify a log session.\n\n"
            f"The agent responded:\n{model_answer}\n\n"
            f"Does the agent's response indicate the logs are "
            f"ANOMALOUS or NORMAL?\n\n"
            f"Respond with exactly: ANOMALY or NORMAL"
        )
        try:
            raw = self._ask_judge(prompt, temperature=0.0, max_tokens=32)
            has_anomaly = bool(_ANOMALY_PATTERN.search(raw))
            predicted = "anomaly" if has_anomaly else "normal"
            reference = record.reference.lower().strip()
            is_correct = predicted == reference
            return is_correct, {
                "match_type": "llm_fallback",
                "predicted": predicted,
                "reference": reference,
                "raw_judge_output": raw,
            }
        except Exception as exc:
            return False, {
                "match_type": "llm_fallback_error",
                "error": str(exc),
            }
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/evals/test_loghub.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/openjarvis/evals/scorers/loghub_scorer.py tests/evals/test_loghub.py
git commit -m "feat(evals): add LogHub scorer for anomaly classification"
```

---

## Task 3: Register LogHub in CLI

**Files:**
- Modify: `src/openjarvis/evals/cli.py`
- Modify: `tests/evals/test_loghub.py`

**Step 1: Write the failing test**

Append to `tests/evals/test_loghub.py`:

```python
class TestLogHubCLI:
    def test_in_benchmarks_dict(self) -> None:
        from openjarvis.evals.cli import BENCHMARKS
        assert "loghub" in BENCHMARKS
        assert BENCHMARKS["loghub"]["category"] == "agentic"

    def test_build_dataset(self) -> None:
        from openjarvis.evals.cli import _build_dataset
        ds = _build_dataset("loghub")
        assert ds is not None
        assert ds.dataset_id == "loghub"

    def test_build_scorer(self) -> None:
        from openjarvis.evals.cli import _build_scorer
        s = _build_scorer("loghub", _mock_backend(), "test-model")
        assert s is not None
        assert s.scorer_id == "loghub"
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/evals/test_loghub.py::TestLogHubCLI -v`
Expected: FAIL with `KeyError` or `ClickException`

**Step 3: Add LogHub to CLI**

Modify `src/openjarvis/evals/cli.py`:

In `BENCHMARKS` dict, add:
```python
"loghub": {"category": "agentic", "description": "LogHub log anomaly detection"},
```

In `_build_dataset()`, add before `else` clause:
```python
elif benchmark == "loghub":
    from openjarvis.evals.datasets.loghub import LogHubDataset
    return LogHubDataset()
```

In `_build_scorer()`, add before `else` clause:
```python
elif benchmark == "loghub":
    from openjarvis.evals.scorers.loghub_scorer import LogHubScorer
    return LogHubScorer(judge_backend, judge_model)
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/evals/test_loghub.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/openjarvis/evals/cli.py tests/evals/test_loghub.py
git commit -m "feat(evals): register LogHub benchmark in CLI"
```

---

## Task 4: AMA-Bench Dataset Provider

**Files:**
- Create: `src/openjarvis/evals/datasets/ama_bench.py`
- Create: `tests/evals/test_ama_bench.py`

**Step 1: Write the failing test**

Create `tests/evals/test_ama_bench.py`:

```python
"""Tests for AMA-Bench dataset provider."""

from openjarvis.evals.datasets.ama_bench import AMABenchDataset


class TestAMABenchDataset:
    def test_instantiation(self) -> None:
        ds = AMABenchDataset()
        assert ds.dataset_id == "ama-bench"
        assert ds.dataset_name == "AMA-Bench"

    def test_has_required_methods(self) -> None:
        ds = AMABenchDataset()
        assert hasattr(ds, "load")
        assert hasattr(ds, "iter_records")
        assert hasattr(ds, "size")
        assert hasattr(ds, "iter_episodes")
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/evals/test_ama_bench.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write minimal implementation**

Create `src/openjarvis/evals/datasets/ama_bench.py`:

```python
"""AMA-Bench: Agent Memory Assessment benchmark.

Evaluates long-horizon agent memory across 4 capability types:
recall, causal inference, state updating, and state abstraction.
Source: https://github.com/AMA-Bench/AMA-Bench
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from openjarvis.evals.core.dataset import DatasetProvider
from openjarvis.evals.core.types import EvalRecord

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are analyzing an agent's interaction trajectory. "
    "The trajectory shows a sequence of actions and observations "
    "from an agent completing a task. "
    "Answer the question about this trajectory accurately and concisely."
)


class AMABenchDataset(DatasetProvider):
    """AMA-Bench agent memory assessment benchmark."""

    dataset_id = "ama-bench"
    dataset_name = "AMA-Bench"

    def __init__(
        self,
        subset: str = "real",
        cache_dir: Optional[str] = None,
    ) -> None:
        self._subset = subset  # "real" or "synthetic"
        self._cache_dir = (
            Path(cache_dir) if cache_dir
            else Path.home() / ".cache" / "ama_bench"
        )
        self._records: List[EvalRecord] = []
        self._episodes: List[List[EvalRecord]] = []

    def load(
        self,
        *,
        max_samples: Optional[int] = None,
        split: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> None:
        data_dir = self._cache_dir / self._subset

        if not data_dir.exists():
            self._download(data_dir)

        trajectories = self._load_trajectories(data_dir)

        if seed is not None:
            random.Random(seed).shuffle(trajectories)
        if max_samples is not None:
            # max_samples applies to trajectories, not individual QA pairs
            trajectories = trajectories[:max_samples]

        self._episodes = []
        self._records = []
        for traj in trajectories:
            episode = self._trajectory_to_episode(traj)
            self._episodes.append(episode)
            self._records.extend(episode)

    def iter_records(self) -> Iterable[EvalRecord]:
        return iter(self._records)

    def iter_episodes(self) -> Iterable[List[EvalRecord]]:
        """Yield grouped QA pairs per trajectory for episode mode."""
        return iter(self._episodes)

    def size(self) -> int:
        return len(self._records)

    def _download(self, data_dir: Path) -> None:
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            raise ImportError(
                "huggingface_hub required. Install with: pip install huggingface_hub"
            )
        data_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id="AMA-Bench/AMA-Bench",
            repo_type="dataset",
            local_dir=str(data_dir),
        )

    def _load_trajectories(
        self, data_dir: Path,
    ) -> List[Dict[str, Any]]:
        """Load trajectory + QA data from disk."""
        trajectories: List[Dict[str, Any]] = []
        # Look for JSON/JSONL files with trajectory data
        for p in sorted(data_dir.rglob("*.json")):
            try:
                with open(p) as f:
                    data = json.load(f)
                if isinstance(data, list):
                    trajectories.extend(data)
                elif isinstance(data, dict):
                    trajectories.append(data)
            except (json.JSONDecodeError, OSError):
                logger.debug("Skipping non-JSON file: %s", p)

        for p in sorted(data_dir.rglob("*.jsonl")):
            try:
                with open(p) as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            trajectories.append(json.loads(line))
            except (json.JSONDecodeError, OSError):
                logger.debug("Skipping non-JSONL file: %s", p)

        return trajectories

    def _trajectory_to_episode(
        self, traj: Dict[str, Any],
    ) -> List[EvalRecord]:
        """Convert a trajectory dict into a list of EvalRecords."""
        traj_id = traj.get("trajectory_id", traj.get("id", "unknown"))
        traj_text = traj.get("trajectory", traj.get("text", ""))
        domain = traj.get("domain", "general")
        qa_pairs = traj.get("qa_pairs", traj.get("questions", []))

        # Truncate very long trajectories for the problem prompt
        if len(traj_text) > 100_000:
            traj_text = traj_text[:100_000] + "\n\n[Trajectory truncated]"

        records: List[EvalRecord] = []
        for i, qa in enumerate(qa_pairs):
            question = qa.get("question", qa.get("q", ""))
            answer = qa.get("answer", qa.get("a", ""))
            capability = qa.get("capability", qa.get("type", "recall"))

            problem = (
                f"{_SYSTEM_PROMPT}\n\n"
                f"## Trajectory\n{traj_text}\n\n"
                f"## Question\n{question}"
            )

            records.append(EvalRecord(
                record_id=f"ama-{traj_id}-q{i}",
                problem=problem,
                reference=answer,
                category="agentic",
                subject=capability,
                metadata={
                    "trajectory_id": traj_id,
                    "domain": domain,
                    "capability": capability,
                    "question_index": i,
                    "trajectory_length": len(traj_text),
                },
            ))

        return records
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/evals/test_ama_bench.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/openjarvis/evals/datasets/ama_bench.py tests/evals/test_ama_bench.py
git commit -m "feat(evals): add AMA-Bench dataset provider with episode support"
```

---

## Task 5: AMA-Bench Scorer + CLI Registration

**Files:**
- Create: `src/openjarvis/evals/scorers/ama_bench_judge.py`
- Modify: `src/openjarvis/evals/cli.py`
- Modify: `tests/evals/test_ama_bench.py`

**Step 1: Write the failing test**

Append to `tests/evals/test_ama_bench.py`:

```python
from unittest.mock import MagicMock
from openjarvis.evals.core.types import EvalRecord
from openjarvis.evals.scorers.ama_bench_judge import AMABenchScorer


def _mock_backend() -> MagicMock:
    backend = MagicMock()
    backend.generate.return_value = "CORRECT"
    return backend


class TestAMABenchScorer:
    def test_instantiation(self) -> None:
        s = AMABenchScorer(_mock_backend(), "test-model")
        assert s.scorer_id == "ama-bench"

    def test_empty_response(self) -> None:
        s = AMABenchScorer(_mock_backend(), "test-model")
        record = EvalRecord(
            record_id="test-1", problem="question",
            reference="answer", category="agentic",
        )
        is_correct, meta = s.score(record, "")
        assert is_correct is False
        assert meta["reason"] == "empty_response"


class TestAMABenchCLI:
    def test_in_benchmarks_dict(self) -> None:
        from openjarvis.evals.cli import BENCHMARKS
        assert "ama-bench" in BENCHMARKS

    def test_build_dataset(self) -> None:
        from openjarvis.evals.cli import _build_dataset
        ds = _build_dataset("ama-bench")
        assert ds.dataset_id == "ama-bench"

    def test_build_scorer(self) -> None:
        from openjarvis.evals.cli import _build_scorer
        s = _build_scorer("ama-bench", _mock_backend(), "test-model")
        assert s.scorer_id == "ama-bench"
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/evals/test_ama_bench.py::TestAMABenchScorer -v`
Expected: FAIL

**Step 3: Implement scorer and register in CLI**

Create `src/openjarvis/evals/scorers/ama_bench_judge.py`:

```python
"""LLM-judge scorer for AMA-Bench agent memory assessment."""

from __future__ import annotations

import re
from typing import Any, Dict, Optional, Tuple

from openjarvis.evals.core.scorer import LLMJudgeScorer
from openjarvis.evals.core.types import EvalRecord

_JUDGE_PROMPT = """You are evaluating an agent memory assessment.

Question: {question}

Reference Answer: {reference}

Agent's Answer: {model_answer}

Is the agent's answer correct? Consider semantic equivalence, not exact wording.
Respond with exactly: CORRECT or INCORRECT"""


class AMABenchScorer(LLMJudgeScorer):
    """Score AMA-Bench QA via LLM judge."""

    scorer_id = "ama-bench"

    def score(
        self, record: EvalRecord, model_answer: str,
    ) -> Tuple[Optional[bool], Dict[str, Any]]:
        if not model_answer or not model_answer.strip():
            return False, {"reason": "empty_response"}

        if not record.reference or not record.reference.strip():
            return None, {"reason": "no_ground_truth"}

        # Extract just the question from the problem (after "## Question")
        question = record.problem
        if "## Question" in question:
            question = question.split("## Question")[-1].strip()

        prompt = _JUDGE_PROMPT.format(
            question=question,
            reference=record.reference,
            model_answer=model_answer,
        )

        try:
            raw = self._ask_judge(prompt, temperature=0.0, max_tokens=64)
            is_correct = bool(re.search(r"\bCORRECT\b", raw, re.IGNORECASE))
            # Check it's not "INCORRECT"
            if re.search(r"\bINCORRECT\b", raw, re.IGNORECASE):
                is_correct = False

            return is_correct, {
                "match_type": "llm_judge",
                "raw_judge_output": raw,
                "capability": record.metadata.get("capability", ""),
            }
        except Exception as exc:
            return False, {
                "match_type": "llm_judge_error",
                "error": str(exc),
            }
```

Modify `src/openjarvis/evals/cli.py` — add to `BENCHMARKS`:
```python
"ama-bench": {"category": "agentic", "description": "AMA-Bench agent memory assessment"},
```

Add to `_build_dataset()`:
```python
elif benchmark == "ama-bench":
    from openjarvis.evals.datasets.ama_bench import AMABenchDataset
    return AMABenchDataset()
```

Add to `_build_scorer()`:
```python
elif benchmark == "ama-bench":
    from openjarvis.evals.scorers.ama_bench_judge import AMABenchScorer
    return AMABenchScorer(judge_backend, judge_model)
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/evals/test_ama_bench.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/openjarvis/evals/scorers/ama_bench_judge.py src/openjarvis/evals/cli.py tests/evals/test_ama_bench.py
git commit -m "feat(evals): add AMA-Bench scorer and CLI registration"
```

---

## Task 6: EvalRunner Episode Mode

**Files:**
- Modify: `src/openjarvis/evals/core/dataset.py`
- Modify: `src/openjarvis/evals/core/types.py`
- Modify: `src/openjarvis/evals/core/runner.py`
- Create: `tests/evals/test_episode_mode.py`

**Step 1: Write the failing test**

Create `tests/evals/test_episode_mode.py`:

```python
"""Tests for EvalRunner episode mode."""

from openjarvis.evals.core.dataset import DatasetProvider
from openjarvis.evals.core.types import EvalRecord


class TestDatasetProviderEpisodes:
    def test_default_iter_episodes(self) -> None:
        """Default iter_episodes wraps each record in its own episode."""

        class SimpleDataset(DatasetProvider):
            dataset_id = "test"
            dataset_name = "Test"

            def __init__(self):
                self._records = [
                    EvalRecord("r1", "q1", "a1", "chat"),
                    EvalRecord("r2", "q2", "a2", "chat"),
                ]

            def load(self, **kw):
                pass

            def iter_records(self):
                return iter(self._records)

            def size(self):
                return len(self._records)

        ds = SimpleDataset()
        episodes = list(ds.iter_episodes())
        assert len(episodes) == 2
        assert len(episodes[0]) == 1
        assert episodes[0][0].record_id == "r1"


class TestRunConfigEpisodeMode:
    def test_episode_mode_field(self) -> None:
        from openjarvis.evals.core.types import RunConfig
        cfg = RunConfig(
            benchmark="test", backend="test", model="test",
            episode_mode=True,
        )
        assert cfg.episode_mode is True

    def test_episode_mode_default_false(self) -> None:
        from openjarvis.evals.core.types import RunConfig
        cfg = RunConfig(benchmark="test", backend="test", model="test")
        assert cfg.episode_mode is False
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/evals/test_episode_mode.py -v`
Expected: FAIL

**Step 3: Implement episode mode**

Add `iter_episodes` default to `src/openjarvis/evals/core/dataset.py`:

```python
def iter_episodes(self) -> Iterable[List["EvalRecord"]]:
    """Iterate over episodes (groups of sequential records).

    Default: each record is its own single-record episode.
    Override for benchmarks requiring sequential processing
    with shared agent state within an episode.
    """
    from openjarvis.evals.core.types import EvalRecord  # noqa: F811
    for record in self.iter_records():
        yield [record]
```

Add `episode_mode` to `RunConfig` in `src/openjarvis/evals/core/types.py`:

```python
episode_mode: bool = False
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/evals/test_episode_mode.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/openjarvis/evals/core/dataset.py src/openjarvis/evals/core/types.py tests/evals/test_episode_mode.py
git commit -m "feat(evals): add episode mode to DatasetProvider and RunConfig"
```

---

## Task 7: EnvironmentProvider ABC

**Files:**
- Create: `src/openjarvis/evals/core/environment.py`
- Create: `tests/evals/test_environment_provider.py`

**Step 1: Write the failing test**

Create `tests/evals/test_environment_provider.py`:

```python
"""Tests for EnvironmentProvider ABC."""

from openjarvis.evals.core.environment import EnvironmentProvider


class _MockEnv(EnvironmentProvider):
    """Concrete implementation for testing."""

    def setup(self):
        return {"url": "http://localhost:8080"}

    def reset(self):
        pass

    def validate(self, record):
        return True, {"status": "ok"}

    def teardown(self):
        pass


class TestEnvironmentProvider:
    def test_concrete_implementation(self) -> None:
        env = _MockEnv()
        info = env.setup()
        assert info["url"] == "http://localhost:8080"

    def test_validate_returns_tuple(self) -> None:
        from openjarvis.evals.core.types import EvalRecord

        env = _MockEnv()
        record = EvalRecord("r1", "problem", "ref", "agentic")
        is_correct, meta = env.validate(record)
        assert is_correct is True
        assert meta["status"] == "ok"

    def test_lifecycle(self) -> None:
        env = _MockEnv()
        env.setup()
        env.reset()
        env.teardown()
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/evals/test_environment_provider.py -v`
Expected: FAIL

**Step 3: Implement**

Create `src/openjarvis/evals/core/environment.py`:

```python
"""Environment provider ABC for benchmarks requiring external environments."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

from openjarvis.evals.core.types import EvalRecord


class EnvironmentProvider(ABC):
    """Manages an external environment for evaluation benchmarks.

    Provides lifecycle management (setup/reset/teardown) and
    environment-state validation for benchmarks that need
    Docker containers, ServiceNow instances, or other live systems.
    """

    @abstractmethod
    def setup(self) -> Dict[str, Any]:
        """Start the environment and return connection info.

        Returns a dict with environment-specific connection details
        (e.g., URLs, ports, credentials).
        """

    @abstractmethod
    def reset(self) -> None:
        """Reset environment state between tasks.

        Called between records within an episode to restore
        the environment to a known state.
        """

    @abstractmethod
    def validate(
        self, record: EvalRecord,
    ) -> Tuple[bool, Dict[str, Any]]:
        """Check environment state against expected outcome.

        Args:
            record: The eval record containing the expected state in metadata.

        Returns:
            (is_correct, metadata) where is_correct indicates whether
            the environment is in the expected state.
        """

    @abstractmethod
    def teardown(self) -> None:
        """Stop the environment and release resources."""
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/evals/test_environment_provider.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/openjarvis/evals/core/environment.py tests/evals/test_environment_provider.py
git commit -m "feat(evals): add EnvironmentProvider ABC for live environments"
```

---

## Task 8: LifelongAgentBench Dataset + Scorer + CLI

**Files:**
- Create: `src/openjarvis/evals/datasets/lifelong_agent.py`
- Create: `src/openjarvis/evals/scorers/lifelong_agent_scorer.py`
- Create: `tests/evals/test_lifelong_agent.py`
- Modify: `src/openjarvis/evals/cli.py`

**Step 1: Write the failing test**

Create `tests/evals/test_lifelong_agent.py`:

```python
"""Tests for LifelongAgentBench benchmark."""

from unittest.mock import MagicMock
from openjarvis.evals.datasets.lifelong_agent import LifelongAgentDataset
from openjarvis.evals.scorers.lifelong_agent_scorer import LifelongAgentScorer
from openjarvis.evals.core.types import EvalRecord


def _mock_backend() -> MagicMock:
    backend = MagicMock()
    backend.generate.return_value = "CORRECT"
    return backend


class TestLifelongAgentDataset:
    def test_instantiation(self) -> None:
        ds = LifelongAgentDataset()
        assert ds.dataset_id == "lifelong-agent"
        assert ds.dataset_name == "LifelongAgentBench"

    def test_has_episode_support(self) -> None:
        ds = LifelongAgentDataset()
        assert hasattr(ds, "iter_episodes")


class TestLifelongAgentScorer:
    def test_instantiation(self) -> None:
        s = LifelongAgentScorer(_mock_backend(), "test-model")
        assert s.scorer_id == "lifelong-agent"

    def test_empty_response(self) -> None:
        s = LifelongAgentScorer(_mock_backend(), "test-model")
        record = EvalRecord("t-1", "task", "expected", "agentic")
        is_correct, meta = s.score(record, "")
        assert is_correct is False


class TestLifelongAgentCLI:
    def test_in_benchmarks(self) -> None:
        from openjarvis.evals.cli import BENCHMARKS
        assert "lifelong-agent" in BENCHMARKS

    def test_build_dataset(self) -> None:
        from openjarvis.evals.cli import _build_dataset
        ds = _build_dataset("lifelong-agent")
        assert ds.dataset_id == "lifelong-agent"

    def test_build_scorer(self) -> None:
        from openjarvis.evals.cli import _build_scorer
        s = _build_scorer("lifelong-agent", _mock_backend(), "test-model")
        assert s.scorer_id == "lifelong-agent"
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/evals/test_lifelong_agent.py -v`
Expected: FAIL

**Step 3: Implement dataset, scorer, and CLI registration**

Create `src/openjarvis/evals/datasets/lifelong_agent.py` — follows same pattern as AMA-Bench but with sequential task episodes across DB/OS/KG environments. Each episode is a list of dependent tasks. `iter_episodes()` yields task sequences.

Create `src/openjarvis/evals/scorers/lifelong_agent_scorer.py` — LLM judge that evaluates whether the agent's action produced the expected outcome. Falls back to exact match for structured outputs.

Register in CLI with key `"lifelong-agent"`.

*(Full implementation code follows the same patterns as Tasks 1-5. The dataset loads from HuggingFace, converts to EvalRecords grouped by episode, and the scorer uses LLM-judge comparison.)*

**Step 4: Run tests**

Run: `uv run pytest tests/evals/test_lifelong_agent.py -v`

**Step 5: Commit**

```bash
git add src/openjarvis/evals/datasets/lifelong_agent.py src/openjarvis/evals/scorers/lifelong_agent_scorer.py src/openjarvis/evals/cli.py tests/evals/test_lifelong_agent.py
git commit -m "feat(evals): add LifelongAgentBench benchmark with episode support"
```

---

## Task 9: WebChoreArena Dataset + Scorer + CLI

**Files:**
- Create: `src/openjarvis/evals/datasets/webchorearena.py`
- Create: `src/openjarvis/evals/scorers/webchorearena_scorer.py`
- Create: `tests/evals/test_webchorearena.py`
- Modify: `src/openjarvis/evals/cli.py`

**Step 1-5:** Same TDD pattern. Dataset loads 532 tasks from WebChoreArena GitHub release. Records include `start_url`, `eval_type` (string_match/url_match/program_html) in metadata. Scorer dispatches to the appropriate evaluator type. Register as `"webchorearena"` in CLI.

**Key implementation details:**
- `WebChoreArenaDataset` loads task JSON configs
- Scorer implements string_match (exact, must_include, fuzzy via LLM), url_match, and program_html evaluation
- Tasks categorized as `massive_memory`, `calculation`, `long_term_memory`

```bash
git commit -m "feat(evals): add WebChoreArena benchmark for web chore tasks"
```

---

## Task 10: WorkArena++ Dataset + Scorer + CLI

**Files:**
- Create: `src/openjarvis/evals/datasets/workarena.py`
- Create: `src/openjarvis/evals/scorers/workarena_scorer.py`
- Create: `tests/evals/test_workarena.py`
- Modify: `src/openjarvis/evals/cli.py`

**Step 1-5:** Same TDD pattern. Dataset loads 682 enterprise workflow tasks from WorkArena GitHub. Records include `level` (L2/L3), `skill` category, `task_type` in metadata. Scorer uses LLM-judge for response quality (environment-state validation requires live ServiceNow — handled by EnvironmentProvider when available, falls back to LLM-judge otherwise). Register as `"workarena"` in CLI.

```bash
git commit -m "feat(evals): add WorkArena++ benchmark for enterprise workflows"
```

---

## Task 11: browser_axtree Tool

**Files:**
- Create: `src/openjarvis/tools/browser_axtree.py`
- Create: `tests/tools/test_browser_axtree.py`

**Step 1: Write the failing test**

Create `tests/tools/test_browser_axtree.py`:

```python
"""Tests for browser_axtree tool."""

from unittest.mock import MagicMock, PropertyMock, patch
from openjarvis.tools.browser_axtree import BrowserAXTreeTool


def _make_mock_page():
    page = MagicMock()
    page.accessibility.snapshot.return_value = {
        "role": "WebArea",
        "name": "Test Page",
        "children": [
            {"role": "heading", "name": "Welcome", "level": 1},
            {"role": "link", "name": "Click me", "url": "https://example.com"},
            {"role": "textbox", "name": "Search", "value": ""},
        ],
    }
    return page


def _make_mock_session(page=None):
    if page is None:
        page = _make_mock_page()
    session = MagicMock()
    type(session).page = PropertyMock(return_value=page)
    return session


class TestBrowserAXTreeTool:
    def test_instantiation(self) -> None:
        tool = BrowserAXTreeTool()
        assert tool.tool_id == "browser_axtree"
        assert tool.spec.name == "browser_axtree"

    def test_execute_returns_tree(self) -> None:
        session = _make_mock_session()
        with patch("openjarvis.tools.browser_axtree._session", session):
            tool = BrowserAXTreeTool()
            result = tool.execute()
        assert result.success is True
        assert "heading" in result.content
        assert "Welcome" in result.content

    def test_playwright_not_installed(self) -> None:
        session = MagicMock()
        type(session).page = PropertyMock(
            side_effect=ImportError("playwright not installed")
        )
        with patch("openjarvis.tools.browser_axtree._session", session):
            tool = BrowserAXTreeTool()
            result = tool.execute()
        assert result.success is False
        assert "playwright" in result.content.lower()
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/tools/test_browser_axtree.py -v`

**Step 3: Implement**

Create `src/openjarvis/tools/browser_axtree.py`:

```python
"""Browser accessibility tree extraction tool.

Extracts the accessibility tree (AX tree) from the current browser page,
providing a structured text representation of the DOM with element IDs,
roles, names, and states. Used by top-performing agents on WebArena-family
benchmarks.
"""

from __future__ import annotations

from typing import Any

from openjarvis.core.registry import ToolRegistry
from openjarvis.core.types import ToolResult
from openjarvis.tools._stubs import BaseTool, ToolSpec


@ToolRegistry.register("browser_axtree")
class BrowserAXTreeTool(BaseTool):
    """Extract the accessibility tree from the current browser page."""

    tool_id = "browser_axtree"

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="browser_axtree",
            description=(
                "Extract the accessibility tree from the current browser page. "
                "Returns a structured text representation with element roles, "
                "names, values, and states. More structured than raw HTML."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "max_depth": {
                        "type": "integer",
                        "description": "Maximum tree depth to traverse. Default: 10.",
                    },
                },
            },
            category="browser",
            required_capabilities=["network:fetch"],
        )

    def execute(self, **params: Any) -> ToolResult:
        max_depth = params.get("max_depth", 10)

        try:
            from openjarvis.tools.browser import _session
            page = _session.page
        except ImportError as exc:
            return ToolResult(
                tool_name="browser_axtree",
                content=f"Playwright not installed: {exc}",
                success=False,
            )

        try:
            snapshot = page.accessibility.snapshot()
            if not snapshot:
                return ToolResult(
                    tool_name="browser_axtree",
                    content="No accessibility tree available.",
                    success=False,
                )

            text = _format_axtree(snapshot, max_depth=max_depth)

            return ToolResult(
                tool_name="browser_axtree",
                content=text,
                success=True,
                metadata={"node_count": _count_nodes(snapshot)},
            )
        except Exception as exc:
            return ToolResult(
                tool_name="browser_axtree",
                content=f"AX tree extraction error: {exc}",
                success=False,
            )


def _format_axtree(
    node: dict, depth: int = 0, max_depth: int = 10,
) -> str:
    """Format an accessibility tree node as indented text."""
    if depth >= max_depth:
        return ""

    indent = "  " * depth
    role = node.get("role", "unknown")
    name = node.get("name", "")
    value = node.get("value", "")

    parts = [f"{indent}[{role}]"]
    if name:
        parts.append(f' "{name}"')
    if value:
        parts.append(f" value={value}")

    line = "".join(parts)
    lines = [line]

    for child in node.get("children", []):
        child_text = _format_axtree(child, depth + 1, max_depth)
        if child_text:
            lines.append(child_text)

    return "\n".join(lines)


def _count_nodes(node: dict) -> int:
    """Count total nodes in the accessibility tree."""
    count = 1
    for child in node.get("children", []):
        count += _count_nodes(child)
    return count
```

**Step 4: Run tests**

Run: `uv run pytest tests/tools/test_browser_axtree.py -v`

**Step 5: Commit**

```bash
git add src/openjarvis/tools/browser_axtree.py tests/tools/test_browser_axtree.py
git commit -m "feat(tools): add browser_axtree tool for accessibility tree extraction"
```

---

## Task 12: MonitorOperativeAgent — Core Structure

**Files:**
- Create: `src/openjarvis/agents/monitor_operative.py`
- Create: `tests/agents/test_monitor_operative.py`

**Step 1: Write the failing test**

Create `tests/agents/test_monitor_operative.py`:

```python
"""Tests for MonitorOperativeAgent."""

from unittest.mock import MagicMock
from openjarvis.agents.monitor_operative import MonitorOperativeAgent
from openjarvis.core.registry import AgentRegistry


def _make_engine(content: str = "Hello") -> MagicMock:
    engine = MagicMock()
    engine.engine_id = "mock"
    engine.generate.return_value = {
        "content": content,
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        "model": "test-model",
        "finish_reason": "stop",
    }
    return engine


class TestMonitorOperativeAgent:
    def test_registration(self) -> None:
        # Import triggers registration
        import openjarvis.agents.monitor_operative  # noqa: F401
        assert AgentRegistry.contains("monitor_operative")

    def test_instantiation(self) -> None:
        engine = _make_engine()
        agent = MonitorOperativeAgent(engine, "test-model")
        assert agent.agent_id == "monitor_operative"
        assert agent.accepts_tools is True

    def test_default_strategies(self) -> None:
        engine = _make_engine()
        agent = MonitorOperativeAgent(engine, "test-model")
        assert agent._memory_extraction == "causality_graph"
        assert agent._observation_compression == "summarize"
        assert agent._retrieval_strategy == "hybrid_with_self_eval"
        assert agent._task_decomposition == "phased"

    def test_custom_strategies(self) -> None:
        engine = _make_engine()
        agent = MonitorOperativeAgent(
            engine, "test-model",
            memory_extraction="scratchpad",
            observation_compression="none",
            retrieval_strategy="keyword",
            task_decomposition="monolithic",
        )
        assert agent._memory_extraction == "scratchpad"
        assert agent._observation_compression == "none"

    def test_simple_run(self) -> None:
        engine = _make_engine("The answer is 42.")
        agent = MonitorOperativeAgent(engine, "test-model")
        result = agent.run("What is the answer?")
        assert result.content == "The answer is 42."
        assert result.turns >= 1
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/agents/test_monitor_operative.py -v`

**Step 3: Implement**

Create `src/openjarvis/agents/monitor_operative.py`:

```python
"""MonitorOperativeAgent — structured memory, observation compression, hybrid retrieval.

Extends OperativeAgent with 4 configurable strategies for long-horizon
monitoring tasks:
- Memory extraction: causality_graph, scratchpad, none
- Observation compression: summarize, axtree, window, none
- Retrieval: hybrid_with_self_eval, hybrid, keyword
- Task decomposition: phased, monolithic
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from openjarvis.agents._stubs import AgentContext, AgentResult, ToolUsingAgent
from openjarvis.core.events import EventBus
from openjarvis.core.registry import AgentRegistry
from openjarvis.core.types import Message, Role, ToolResult
from openjarvis.engine._stubs import InferenceEngine
from openjarvis.tools._stubs import BaseTool

logger = logging.getLogger(__name__)

_VALID_MEMORY_EXTRACTION = {"causality_graph", "scratchpad", "none"}
_VALID_OBSERVATION_COMPRESSION = {"summarize", "axtree", "window", "none"}
_VALID_RETRIEVAL = {"hybrid_with_self_eval", "hybrid", "keyword"}
_VALID_TASK_DECOMPOSITION = {"phased", "monolithic"}


@AgentRegistry.register("monitor_operative")
class MonitorOperativeAgent(ToolUsingAgent):
    """Persistent monitoring agent with structured memory and strategies."""

    agent_id = "monitor_operative"
    accepts_tools = True

    def __init__(
        self,
        engine: InferenceEngine,
        model: str,
        *,
        tools: Optional[List[BaseTool]] = None,
        bus: Optional[EventBus] = None,
        max_turns: int = 30,
        temperature: float = 0.2,
        max_tokens: int = 2048,
        system_prompt: Optional[str] = None,
        operator_id: Optional[str] = None,
        session_store: Optional[Any] = None,
        memory_backend: Optional[Any] = None,
        kg_memory: Optional[Any] = None,
        # Strategy configuration
        memory_extraction: str = "causality_graph",
        observation_compression: str = "summarize",
        retrieval_strategy: str = "hybrid_with_self_eval",
        task_decomposition: str = "phased",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            engine, model, tools=tools, bus=bus,
            max_turns=max_turns, temperature=temperature,
            max_tokens=max_tokens,
        )
        self._system_prompt = system_prompt or ""
        self._operator_id = operator_id
        self._session_store = session_store
        self._memory_backend = memory_backend
        self._kg_memory = kg_memory

        # Validate and set strategies
        if memory_extraction not in _VALID_MEMORY_EXTRACTION:
            raise ValueError(f"Invalid memory_extraction: {memory_extraction}")
        if observation_compression not in _VALID_OBSERVATION_COMPRESSION:
            raise ValueError(f"Invalid observation_compression: {observation_compression}")
        if retrieval_strategy not in _VALID_RETRIEVAL:
            raise ValueError(f"Invalid retrieval_strategy: {retrieval_strategy}")
        if task_decomposition not in _VALID_TASK_DECOMPOSITION:
            raise ValueError(f"Invalid task_decomposition: {task_decomposition}")

        self._memory_extraction = memory_extraction
        self._observation_compression = observation_compression
        self._retrieval_strategy = retrieval_strategy
        self._task_decomposition = task_decomposition

    # -- State management (from OperativeAgent pattern) --

    def _recall_state(self) -> str:
        if not self._memory_backend or not self._operator_id:
            return ""
        state_key = f"operator:{self._operator_id}:state"
        try:
            result = self._memory_backend.retrieve(state_key)
            if result:
                return result if isinstance(result, str) else str(result)
        except Exception:
            logger.debug("No previous state for %s", self._operator_id)
        return ""

    def _load_session(self) -> List[Message]:
        if not self._session_store or not self._operator_id:
            return []
        session_id = f"operator:{self._operator_id}"
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
            logger.debug("Could not load session for %s", self._operator_id)
        return []

    def _save_session(self, input_text: str, response: str) -> None:
        if not self._session_store or not self._operator_id:
            return
        session_id = f"operator:{self._operator_id}"
        try:
            self._session_store.save_message(
                session_id, {"role": "user", "content": input_text},
            )
            self._session_store.save_message(
                session_id, {"role": "assistant", "content": response},
            )
        except Exception:
            logger.debug("Could not save session for %s", self._operator_id)

    def _auto_persist_state(self, content: str) -> None:
        if not self._memory_backend or not self._operator_id:
            return
        state_key = f"operator:{self._operator_id}:state"
        try:
            summary = content[:1000] if content else ""
            self._memory_backend.store(state_key, summary)
        except Exception:
            logger.debug("Could not auto-persist state for %s", self._operator_id)

    # -- Observation compression --

    def _compress_observation(self, observation: str) -> str:
        """Compress a tool observation based on strategy."""
        if self._observation_compression == "none":
            return observation
        if len(observation) <= 2000:
            return observation

        if self._observation_compression == "summarize":
            return self._summarize_observation(observation)
        elif self._observation_compression == "window":
            # Keep first and last portions
            half = 1000
            return (
                observation[:half]
                + "\n\n[... content truncated ...]\n\n"
                + observation[-half:]
            )
        elif self._observation_compression == "axtree":
            # AXTree observations are already structured — just truncate
            lines = observation.split("\n")
            if len(lines) > 100:
                return "\n".join(lines[:100]) + "\n[... truncated ...]"
            return observation

        return observation

    def _summarize_observation(self, observation: str) -> str:
        """Summarize a long observation using the LLM."""
        try:
            prompt = (
                "Summarize the following observation concisely, "
                "preserving key facts and data:\n\n"
                + observation[:8000]
            )
            messages = [Message(role=Role.USER, content=prompt)]
            result = self._generate(messages)
            return result.get("content", observation[:2000])
        except Exception:
            return observation[:2000]

    # -- Memory extraction --

    def _extract_and_store(
        self, tool_name: str, tool_result: str,
    ) -> None:
        """Extract entities/relations from a tool result and store."""
        if self._memory_extraction == "none":
            return
        if self._memory_extraction == "scratchpad":
            return  # Agent manages via memory_store tool
        if self._memory_extraction == "causality_graph":
            self._extract_causality(tool_name, tool_result)

    def _extract_causality(
        self, tool_name: str, tool_result: str,
    ) -> None:
        """Extract causal entities and relations from a tool result."""
        if not self._kg_memory:
            return
        if len(tool_result) < 10:
            return

        try:
            prompt = (
                "Extract key entities and causal relations from this tool result.\n"
                f"Tool: {tool_name}\n"
                f"Result: {tool_result[:4000]}\n\n"
                "Output as JSON:\n"
                '{"entities": [{"id": "...", "type": "...", "name": "..."}], '
                '"relations": [{"source": "...", "target": "...", "type": "..."}]}'
            )
            messages = [Message(role=Role.USER, content=prompt)]
            result = self._generate(messages)
            content = result.get("content", "")

            # Parse JSON from response
            import re
            json_match = re.search(r"\{.*\}", content, re.DOTALL)
            if not json_match:
                return

            data = json.loads(json_match.group())

            from openjarvis.tools.storage.knowledge_graph import Entity, Relation

            for ent in data.get("entities", []):
                self._kg_memory.add_entity(Entity(
                    entity_id=ent.get("id", ""),
                    entity_type=ent.get("type", "concept"),
                    name=ent.get("name", ""),
                ))

            for rel in data.get("relations", []):
                self._kg_memory.add_relation(Relation(
                    source_id=rel.get("source", ""),
                    target_id=rel.get("target", ""),
                    relation_type=rel.get("type", "related"),
                ))
        except Exception:
            logger.debug("Causality extraction failed for %s", tool_name)

    # -- Main run loop --

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

        previous_state = self._recall_state()
        if previous_state:
            sys_parts.append(f"\n## Previous State\n{previous_state}")

        system_prompt = "\n\n".join(sys_parts) if sys_parts else None

        # Load session history
        session_messages = self._load_session()

        # Build messages
        messages = self._build_messages(input, context, system_prompt=system_prompt)
        # Prepend session messages after system message
        if session_messages:
            messages = [messages[0]] + session_messages + messages[1:]

        # Tool-calling loop
        openai_tools = self._executor.get_openai_tools() if self._tools else []
        all_tool_results: List[ToolResult] = []
        turns = 0
        content = ""
        state_stored = False

        for _turn in range(self._max_turns):
            turns += 1

            gen_kwargs: Dict[str, Any] = {}
            if openai_tools:
                gen_kwargs["tools"] = openai_tools

            result = self._generate(messages, **gen_kwargs)
            content = result.get("content", "")
            raw_tool_calls = result.get("tool_calls", [])

            if not raw_tool_calls:
                content = self._check_continuation(result, messages)
                break

            # Build ToolCall objects
            from openjarvis.tools._stubs import ToolCall
            tool_calls = [
                ToolCall(
                    id=tc.get("id", f"call_{i}"),
                    name=tc.get("name", ""),
                    arguments=tc.get("arguments", "{}"),
                )
                for i, tc in enumerate(raw_tool_calls)
            ]

            messages.append(Message(
                role=Role.ASSISTANT, content=content, tool_calls=tool_calls,
            ))

            for tc in tool_calls:
                # Loop guard check
                if self._loop_guard:
                    verdict = self._loop_guard.check_call(tc.name, tc.arguments)
                    if verdict.blocked:
                        tr = ToolResult(
                            tool_name=tc.name,
                            content=f"Loop guard: {verdict.reason}",
                            success=False,
                        )
                        all_tool_results.append(tr)
                        messages.append(Message(
                            role=Role.TOOL, content=tr.content,
                            tool_call_id=tc.id, name=tc.name,
                        ))
                        continue

                tr = self._executor.execute(tc)
                all_tool_results.append(tr)

                # Compress observation
                compressed = self._compress_observation(tr.content)

                # Extract and store in KG
                self._extract_and_store(tc.name, tr.content)

                # Track if state was stored
                if tc.name == "memory_store" and self._operator_id:
                    state_stored = True

                messages.append(Message(
                    role=Role.TOOL, content=compressed,
                    tool_call_id=tc.id, name=tc.name,
                ))

        # Save session and auto-persist state
        self._save_session(input, content)
        if not state_stored:
            self._auto_persist_state(content)

        self._emit_turn_end(turns=turns, content_length=len(content))
        return AgentResult(
            content=content,
            tool_results=all_tool_results,
            turns=turns,
        )
```

**Step 4: Run tests**

Run: `uv run pytest tests/agents/test_monitor_operative.py -v`

**Step 5: Commit**

```bash
git add src/openjarvis/agents/monitor_operative.py tests/agents/test_monitor_operative.py
git commit -m "feat(agents): add MonitorOperativeAgent with configurable strategies"
```

---

## Task 13: Register MonitorOperativeAgent in Agent Package

**Files:**
- Modify: `src/openjarvis/agents/__init__.py`

**Step 1: Add import**

Add to `src/openjarvis/agents/__init__.py`:

```python
try:
    import openjarvis.agents.monitor_operative  # noqa: F401
except ImportError:
    pass
```

**Step 2: Run full agent test suite**

Run: `uv run pytest tests/agents/ -v`
Expected: PASS

**Step 3: Commit**

```bash
git add src/openjarvis/agents/__init__.py
git commit -m "feat(agents): register MonitorOperativeAgent in agent package"
```

---

## Task 14: Operator Recipe for MonitorOperativeAgent

**Files:**
- Create: `src/openjarvis/recipes/data/operators/monitor.toml`

**Step 1: Create recipe**

```toml
[operator]
name = "monitor"
description = "Long-horizon monitoring agent with structured memory extraction, observation compression, hybrid retrieval, and phased task decomposition."

[operator.agent]
type = "monitor_operative"
max_turns = 30
temperature = 0.2
tools = [
    "think", "calculator", "code_interpreter",
    "memory_store", "memory_search", "memory_retrieve",
    "kg_add_entity", "kg_add_relation", "kg_query", "kg_neighbors",
    "browser_navigate", "browser_click", "browser_type",
    "browser_extract", "browser_screenshot", "browser_axtree",
    "file_read", "file_write", "web_search", "http_request",
    "shell_exec", "db_query",
]

[operator.strategies]
memory_extraction = "causality_graph"
observation_compression = "summarize"
retrieval = "hybrid_with_self_eval"
task_decomposition = "phased"

[operator.schedule]
type = "cron"
value = "0 */2 * * *"
```

**Step 2: Commit**

```bash
git add src/openjarvis/recipes/data/operators/monitor.toml
git commit -m "feat(recipes): add monitor operator recipe"
```

---

## Task 15: Run Full Test Suite

**Step 1: Run all tests**

Run: `uv run pytest tests/ -v --tb=short 2>&1 | tail -20`
Expected: All existing tests + new tests PASS

**Step 2: Run lint**

Run: `uv run ruff check src/openjarvis/evals/datasets/loghub.py src/openjarvis/evals/scorers/loghub_scorer.py src/openjarvis/evals/datasets/ama_bench.py src/openjarvis/evals/scorers/ama_bench_judge.py src/openjarvis/evals/core/environment.py src/openjarvis/agents/monitor_operative.py src/openjarvis/tools/browser_axtree.py`

Fix any lint issues.

**Step 3: Commit fixes if any**

```bash
git commit -m "fix: lint and test fixes for operator benchmarks"
```

---

## Task 16: Update Documentation

**Files:**
- Modify: `CLAUDE.md`

**Step 1: Add Phase 25 to CLAUDE.md**

Add to the `## Project Status` section — update test count and phase number.

Add to `## Architecture` — under existing eval framework section, mention the 5 new benchmarks.

Add to `## Development Phases` table:
```
| v3.0 | 25 | Operator benchmarks: LogHub, AMA-Bench, LifelongAgentBench, WebChoreArena, WorkArena++. MonitorOperativeAgent with 4 strategies. EvalRunner episode mode. EnvironmentProvider ABC. browser_axtree tool |
```

Add to `## API Surface` — add the 5 new benchmark keys to the eval list.

**Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md for Phase 25 operator benchmarks"
```

---

## Task 17: Sanity Check — LogHub (5 queries)

**Step 1: Run LogHub evaluation**

Run: `uv run jarvis eval run -b loghub -m "qwen3:32b" -n 5 --backend jarvis-agent -v`

Expected output: 5 samples evaluated with pass/fail results and accuracy score.

**Step 2: Verify output files exist**

Check: `ls results/loghub_qwen3-32b.jsonl results/loghub_qwen3-32b.summary.json`

---

## Task 18: Sanity Check — AMA-Bench (5 queries)

Run: `uv run jarvis eval run -b ama-bench -m "qwen3:32b" -n 5 --backend jarvis-agent -v`

---

## Task 19: Sanity Check — LifelongAgentBench (5 queries)

Run: `uv run jarvis eval run -b lifelong-agent -m "qwen3:32b" -n 5 --backend jarvis-agent -v`

---

## Task 20: Sanity Check — WebChoreArena (5 queries)

Run: `uv run jarvis eval run -b webchorearena -m "qwen3:32b" -n 5 --backend jarvis-agent -v`

Note: Requires WebArena Docker stack running.

---

## Task 21: Sanity Check — WorkArena++ (5 queries)

Run: `uv run jarvis eval run -b workarena -m "qwen3:32b" -n 5 --backend jarvis-agent -v`

Note: Requires ServiceNow developer instance with SERVICENOW_* env vars.
