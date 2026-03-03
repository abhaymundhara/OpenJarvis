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
