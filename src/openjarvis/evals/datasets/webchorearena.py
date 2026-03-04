"""WebChoreArena: Web chore task benchmark.

532 tasks on 4 Docker web environments testing massive memory,
calculation, and long-term memory capabilities.
Source: https://github.com/WebChoreArena/WebChoreArena
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
    "You are a web automation agent. Complete the following task "
    "using browser tools to interact with the web application. "
    "Provide your final answer clearly."
)

_EVAL_TYPES = {"string_match", "url_match", "program_html"}


class WebChoreArenaDataset(DatasetProvider):
    """WebChoreArena web chore task benchmark."""

    dataset_id = "webchorearena"
    dataset_name = "WebChoreArena"

    def __init__(
        self,
        cache_dir: Optional[str] = None,
    ) -> None:
        self._cache_dir = (
            Path(cache_dir) if cache_dir
            else Path.home() / ".cache" / "webchorearena"
        )
        self._records: List[EvalRecord] = []

    def load(
        self,
        *,
        max_samples: Optional[int] = None,
        split: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> None:
        data_dir = self._cache_dir

        if not data_dir.exists():
            self._download(data_dir)

        records = self._load_tasks(data_dir)

        if seed is not None:
            random.Random(seed).shuffle(records)
        if max_samples is not None:
            records = records[:max_samples]

        self._records = records

    def iter_records(self) -> Iterable[EvalRecord]:
        return iter(self._records)

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
            repo_id="WebChoreArena/WebChoreArena",
            repo_type="dataset",
            local_dir=str(data_dir),
        )

    def _load_tasks(self, data_dir: Path) -> List[EvalRecord]:
        """Load task configs from JSON files."""
        records: List[EvalRecord] = []

        for p in sorted(data_dir.rglob("*.json")):
            try:
                with open(p) as f:
                    data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        rec = self._task_to_record(item)
                        if rec:
                            records.append(rec)
                elif isinstance(data, dict):
                    rec = self._task_to_record(data)
                    if rec:
                        records.append(rec)
            except (json.JSONDecodeError, OSError):
                logger.debug("Skipping file: %s", p)

        for p in sorted(data_dir.rglob("*.jsonl")):
            try:
                with open(p) as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            item = json.loads(line)
                            rec = self._task_to_record(item)
                            if rec:
                                records.append(rec)
            except (json.JSONDecodeError, OSError):
                logger.debug("Skipping file: %s", p)

        return records

    def _task_to_record(self, task: Dict[str, Any]) -> Optional[EvalRecord]:
        """Convert a task dict to an EvalRecord."""
        task_id = task.get("task_id", task.get("id", ""))
        if not task_id:
            return None

        intent = task.get("intent", task.get("task", task.get("instruction", "")))
        reference = task.get(
            "reference_answer", task.get("answer", task.get("expected", "")),
        )
        eval_type = task.get(
            "eval_type", task.get("evaluation_type", "string_match"),
        )
        start_url = task.get("start_url", task.get("url", ""))
        category = task.get("category", task.get("chore_type", "massive_memory"))
        sites = task.get("sites", [])

        problem = f"{_SYSTEM_PROMPT}\n\n## Task\n{intent}"
        if start_url:
            problem += f"\n\nStart URL: {start_url}"

        return EvalRecord(
            record_id=f"wca-{task_id}",
            problem=problem,
            reference=str(reference),
            category="agentic",
            subject=category,
            metadata={
                "task_id": task_id,
                "eval_type": eval_type,
                "start_url": start_url,
                "sites": sites,
                "category": category,
            },
        )


__all__ = ["WebChoreArenaDataset"]
