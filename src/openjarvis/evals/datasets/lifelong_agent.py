"""LifelongAgentBench: Sequential task learning benchmark.

Evaluates agents on sequential tasks across DB, OS, and KG environments
where knowledge from previous tasks is needed.
Source: arXiv:2505.11942
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
    "You are an agent operating in a live environment. "
    "Complete the following task using the tools available to you. "
    "Previous tasks in this session may have modified the environment."
)


class LifelongAgentDataset(DatasetProvider):
    """LifelongAgentBench sequential task learning benchmark."""

    dataset_id = "lifelong-agent"
    dataset_name = "LifelongAgentBench"

    def __init__(
        self,
        subset: str = "database",
        cache_dir: Optional[str] = None,
    ) -> None:
        self._subset = subset  # "database", "os", "knowledge_graph"
        self._cache_dir = (
            Path(cache_dir) if cache_dir
            else Path.home() / ".cache" / "lifelong_agent"
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

        task_sequences = self._load_task_sequences(data_dir)

        if seed is not None:
            random.Random(seed).shuffle(task_sequences)
        if max_samples is not None:
            task_sequences = task_sequences[:max_samples]

        self._episodes = []
        self._records = []
        for seq in task_sequences:
            episode = self._sequence_to_episode(seq)
            self._episodes.append(episode)
            self._records.extend(episode)

    def iter_records(self) -> Iterable[EvalRecord]:
        return iter(self._records)

    def iter_episodes(self) -> Iterable[List[EvalRecord]]:
        """Yield task sequences for episode mode."""
        return iter(self._episodes)

    def size(self) -> int:
        return len(self._records)

    def _download(self, data_dir: Path) -> None:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise ImportError(
                "huggingface_hub required. Install with: pip install huggingface_hub"
            ) from exc
        data_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id="LifelongAgentBench/LifelongAgentBench",
            repo_type="dataset",
            local_dir=str(data_dir),
        )

    def _load_task_sequences(
        self, data_dir: Path,
    ) -> List[List[Dict[str, Any]]]:
        """Load task sequences from disk."""
        sequences: List[List[Dict[str, Any]]] = []

        for p in sorted(data_dir.rglob("*.json")):
            try:
                with open(p) as f:
                    data = json.load(f)
                if isinstance(data, list):
                    # Could be a sequence of tasks or list of sequences
                    if data and isinstance(data[0], list):
                        sequences.extend(data)
                    elif data and isinstance(data[0], dict):
                        sequences.append(data)
                elif isinstance(data, dict):
                    tasks = data.get("tasks", data.get("sequence", [data]))
                    if tasks:
                        sequences.append(tasks if isinstance(tasks, list) else [tasks])
            except (json.JSONDecodeError, OSError):
                logger.debug("Skipping file: %s", p)

        for p in sorted(data_dir.rglob("*.jsonl")):
            try:
                seq: List[Dict[str, Any]] = []
                with open(p) as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            seq.append(json.loads(line))
                if seq:
                    sequences.append(seq)
            except (json.JSONDecodeError, OSError):
                logger.debug("Skipping file: %s", p)

        return sequences

    def _sequence_to_episode(
        self, tasks: List[Dict[str, Any]],
    ) -> List[EvalRecord]:
        """Convert a task sequence into EvalRecords."""
        records: List[EvalRecord] = []
        seq_id = tasks[0].get("sequence_id", tasks[0].get("id", "unknown")) if tasks else "unknown"

        for i, task in enumerate(tasks):
            task_id = task.get("task_id", task.get("id", f"task-{i}"))
            instruction = task.get("instruction", task.get("task", ""))
            expected = task.get("expected_output", task.get("answer", ""))
            env_type = task.get("environment", self._subset)
            dependencies = task.get("dependencies", [])

            problem = (
                f"{_SYSTEM_PROMPT}\n\n"
                f"## Task {i + 1}\n{instruction}"
            )
            if dependencies:
                problem += f"\n\nThis task depends on previous tasks: {dependencies}"

            records.append(EvalRecord(
                record_id=f"lifelong-{seq_id}-t{i}",
                problem=problem,
                reference=expected,
                category="agentic",
                subject=env_type,
                metadata={
                    "sequence_id": seq_id,
                    "task_index": i,
                    "environment": env_type,
                    "dependencies": dependencies,
                    "task_id": task_id,
                },
            ))

        return records


__all__ = ["LifelongAgentDataset"]
