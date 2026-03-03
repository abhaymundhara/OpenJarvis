"""LLM-judge scorer for LifelongAgentBench."""

from __future__ import annotations

import re
from typing import Any, Dict, Optional, Tuple

from openjarvis.evals.core.scorer import LLMJudgeScorer
from openjarvis.evals.core.types import EvalRecord

_JUDGE_PROMPT = """You are evaluating whether an agent completed a task correctly.

Task: {task}

Expected Outcome: {reference}

Agent's Response: {model_answer}

Did the agent produce the correct result? Consider semantic equivalence.
Respond with exactly: CORRECT or INCORRECT"""


class LifelongAgentScorer(LLMJudgeScorer):
    """Score LifelongAgentBench tasks via LLM judge."""

    scorer_id = "lifelong-agent"

    def score(
        self, record: EvalRecord, model_answer: str,
    ) -> Tuple[Optional[bool], Dict[str, Any]]:
        if not model_answer or not model_answer.strip():
            return False, {"reason": "empty_response"}

        if not record.reference or not record.reference.strip():
            return None, {"reason": "no_ground_truth"}

        task = record.problem
        if "## Task" in task:
            task = task.split("## Task")[-1].strip()

        prompt = _JUDGE_PROMPT.format(
            task=task,
            reference=record.reference,
            model_answer=model_answer,
        )

        try:
            raw = self._ask_judge(prompt, temperature=0.0, max_tokens=64)
            is_correct = bool(re.search(r"\bCORRECT\b", raw, re.IGNORECASE))
            if re.search(r"\bINCORRECT\b", raw, re.IGNORECASE):
                is_correct = False

            return is_correct, {
                "match_type": "llm_judge",
                "raw_judge_output": raw,
                "environment": record.metadata.get("environment", ""),
                "task_index": record.metadata.get("task_index", -1),
            }
        except Exception as exc:
            return False, {
                "match_type": "llm_judge_error",
                "error": str(exc),
            }
