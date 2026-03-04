"""Scorer for WebChoreArena web chore tasks."""

from __future__ import annotations

import re
from typing import Any, Dict, Optional, Tuple

from openjarvis.evals.core.scorer import LLMJudgeScorer
from openjarvis.evals.core.types import EvalRecord


class WebChoreArenaScorer(LLMJudgeScorer):
    """Score WebChoreArena tasks with multiple evaluation types."""

    scorer_id = "webchorearena"

    def score(
        self, record: EvalRecord, model_answer: str,
    ) -> Tuple[Optional[bool], Dict[str, Any]]:
        if not model_answer or not model_answer.strip():
            return False, {"reason": "empty_response"}

        eval_type = record.metadata.get("eval_type", "string_match")
        reference = record.reference

        if eval_type == "string_match":
            return self._string_match(reference, model_answer)
        elif eval_type == "url_match":
            return self._url_match(reference, model_answer)
        elif eval_type == "program_html":
            return self._llm_judge_eval(record, model_answer)
        else:
            return self._llm_judge_eval(record, model_answer)

    def _string_match(
        self, reference: str, model_answer: str,
    ) -> Tuple[bool, Dict[str, Any]]:
        """Exact or fuzzy string matching."""
        ref_clean = reference.strip().lower()
        ans_clean = model_answer.strip().lower()

        # Exact match
        if ref_clean == ans_clean:
            return True, {"match_type": "exact_string", "eval_type": "string_match"}

        # Check if reference is contained in answer
        if ref_clean in ans_clean:
            return True, {"match_type": "contains", "eval_type": "string_match"}

        return False, {
            "match_type": "string_mismatch",
            "eval_type": "string_match",
            "reference": reference,
            "answer_preview": model_answer[:200],
        }

    def _url_match(
        self, reference: str, model_answer: str,
    ) -> Tuple[bool, Dict[str, Any]]:
        """URL matching -- normalize and compare."""
        ref_url = reference.strip().rstrip("/").lower()
        # Extract URL from answer if present
        url_pattern = re.compile(r"https?://[^\s\"'<>]+")
        urls = url_pattern.findall(model_answer)

        for url in urls:
            clean = url.strip().rstrip("/").lower()
            if clean == ref_url or ref_url in clean:
                return True, {"match_type": "url_match", "matched_url": url}

        # Also check plain text match
        if ref_url in model_answer.strip().rstrip("/").lower():
            return True, {"match_type": "url_text_match"}

        return False, {
            "match_type": "url_mismatch",
            "eval_type": "url_match",
            "reference": reference,
        }

    def _llm_judge_eval(
        self, record: EvalRecord, model_answer: str,
    ) -> Tuple[Optional[bool], Dict[str, Any]]:
        """Use LLM judge for program_html and complex evaluations."""
        prompt = (
            f"Evaluate whether the agent completed this web task correctly.\n\n"
            f"Task: {record.problem[:2000]}\n\n"
            f"Expected: {record.reference}\n\n"
            f"Agent's result: {model_answer[:2000]}\n\n"
            f"Did the agent produce the correct result? "
            f"Respond with exactly: CORRECT or INCORRECT"
        )
        try:
            raw = self._ask_judge(prompt, temperature=0.0, max_tokens=64)
            is_correct = bool(re.search(r"\bCORRECT\b", raw, re.IGNORECASE))
            if re.search(r"\bINCORRECT\b", raw, re.IGNORECASE):
                is_correct = False
            return is_correct, {
                "match_type": "llm_judge",
                "eval_type": record.metadata.get("eval_type", "unknown"),
                "raw_judge_output": raw,
            }
        except Exception as exc:
            return False, {"match_type": "llm_judge_error", "error": str(exc)}


__all__ = ["WebChoreArenaScorer"]
