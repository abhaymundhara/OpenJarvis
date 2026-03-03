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
        except ImportError as exc:
            raise ImportError(
                "huggingface_hub required. Install with: pip install huggingface_hub"
            ) from exc
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


__all__ = ["AMABenchDataset"]
