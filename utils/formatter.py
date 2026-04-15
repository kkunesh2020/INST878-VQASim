"""Formatting and persistence helpers for pipeline outputs."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any


TASK_TYPE_LABELS = {
    "reading",
    "description",
    "identification",
    "localization",
    "verification",
    "combination",
    "failure-aware",
    "failure aware",
}


def _normalize_text(value: str) -> str:
    return " ".join((value or "").strip().lower().replace("_", " ").split())


def _tokenize(value: str) -> set[str]:
    tokens = []
    for token in _normalize_text(value).replace(",", " ").replace("/", " ").split():
        cleaned = "".join(ch for ch in token if ch.isalnum())
        if cleaned:
            tokens.append(cleaned)
    return set(tokens)


def _parse_task_type_set(value: str) -> set[str]:
    normalized = _normalize_text(value)
    if not normalized:
        return set()

    parts = [part.strip() for part in normalized.replace(" and ", ",").split(",") if part.strip()]
    labels: set[str] = set()
    for part in parts:
        if part in TASK_TYPE_LABELS:
            labels.add(part.replace(" ", "-"))
    return labels


def _score_target_match(generated_target: str, ground_truth_target: str) -> float:
    gen = _normalize_text(generated_target)
    gt = _normalize_text(ground_truth_target)

    if not gen or not gt:
        return 0.0
    if gen == gt:
        return 1.0
    if gen in gt or gt in gen:
        return 0.5

    gen_tokens = _tokenize(gen)
    gt_tokens = _tokenize(gt)
    if not gen_tokens or not gt_tokens:
        return 0.0

    overlap = len(gen_tokens & gt_tokens)
    if overlap == 0:
        return 0.0

    union = len(gen_tokens | gt_tokens)
    jaccard = overlap / union if union else 0.0
    return 0.5 if jaccard >= 0.35 else 0.0


def _score_task_type_match(generated_type: str, ground_truth_type: str) -> float:
    gen_set = _parse_task_type_set(generated_type)
    gt_set = _parse_task_type_set(ground_truth_type)

    if not gen_set or not gt_set:
        return 0.0
    if gen_set == gt_set:
        return 1.0
    if gen_set & gt_set:
        return 0.5
    return 0.0


def _build_candidate_lookup(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sortable: list[tuple[int, dict[str, Any]]] = []
    for index, candidate in enumerate(candidates):
        rank = candidate.get("rank")
        try:
            sort_rank = int(rank)
        except Exception:
            sort_rank = index + 1
        sortable.append((sort_rank, candidate))
    sortable.sort(key=lambda item: item[0])
    return [item[1] for item in sortable]


def _quantize_score(score: float) -> int | float:
    if score >= 1.0:
        return 1
    if score >= 0.5:
        return 0.5
    return 0


def _extract_question_fields(question: Any) -> tuple[str, str, str]:
    """Best-effort extraction of question text, response target, and task type."""

    if isinstance(question, dict):
        question_text = str(question.get("response") or question.get("question") or question.get("text") or "")
        response_target = str(question.get("response_target", "") or "")
        task_type = str(question.get("task_type", "") or "")
        return question_text, response_target, task_type

    if isinstance(question, str):
        parsed: Any = None
        try:
            parsed = ast.literal_eval(question)
        except (ValueError, SyntaxError):
            parsed = None

        if isinstance(parsed, dict):
            question_text = str(parsed.get("response") or parsed.get("question") or parsed.get("text") or question)
            response_target = str(parsed.get("response_target", "") or "")
            task_type = str(parsed.get("task_type", "") or "")
            return question_text, response_target, task_type

    return str(question), "", ""


def build_comparison(
    question_payload: dict[str, Any],
    ground_truth_question: str,
    response_target_payload: dict[str, Any],
) -> dict[str, Any]:
    """Create per-question target/type match scores against a ground-truth question."""

    ranked_questions = question_payload.get("ranked_questions", [])
    candidates = question_payload.get("candidates", [])
    if not isinstance(ranked_questions, list):
        ranked_questions = []
    if not isinstance(candidates, list):
        candidates = []

    sorted_candidates = _build_candidate_lookup([item for item in candidates if isinstance(item, dict)])
    gt_target = str(response_target_payload.get("response_target", "") or "")
    gt_task_type = str(response_target_payload.get("task_type", "") or "")

    per_question_scores: list[dict[str, Any]] = []
    for index, question in enumerate(ranked_questions, start=1):
        candidate = sorted_candidates[index - 1] if index - 1 < len(sorted_candidates) else {}
        question_text, fallback_target, fallback_task_type = _extract_question_fields(question)

        generated_target = str(candidate.get("response_target", "") or fallback_target or "")
        generated_task_type = str(candidate.get("task_type", "") or fallback_task_type or "")

        target_match_score = _quantize_score(_score_target_match(generated_target, gt_target))
        task_type_match_score = _quantize_score(_score_task_type_match(generated_task_type, gt_task_type))

        per_question_scores.append(
            {
                "rank": index,
                "generated_question": question_text,
                "generated_response_target": generated_target,
                "generated_task_type": generated_task_type,
                "ground_truth_response_target": gt_target,
                "ground_truth_task_type": gt_task_type,
                "target_match_score": target_match_score,
                "task_type_match_score": task_type_match_score,
            }
        )

    per_question_scores.append(
        {
            "rank": "ground_truth",
            "generated_question": str(ground_truth_question),
            "generated_response_target": gt_target,
            "generated_task_type": gt_task_type,
            "ground_truth_response_target": gt_target,
            "ground_truth_task_type": gt_task_type,
            "target_match_score": 1,
            "task_type_match_score": 1,
        }
    )

    return {
        "ground_truth": ground_truth_question,
        "ground_truth_response_target": gt_target,
        "ground_truth_task_type": gt_task_type,
        "per_question_scores": per_question_scores,
    }


def save_json_output(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON output to disk."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=True)


def _render_contexts(contexts_payload: dict[str, Any]) -> list[str]:
    contexts = contexts_payload.get("contexts", [])
    if not isinstance(contexts, list) or not contexts:
        return ["No contexts generated."]

    lines: list[str] = []
    for index, item in enumerate(contexts, start=1):
        if isinstance(item, dict):
            lines.append(f"{index}.")
            lines.append(f"context_id: {item.get('context_id', '')}")
            lines.append(f"location: {item.get('location', '')}")
            lines.append(f"time_pressure: {item.get('time_pressure', '')}")
            lines.append(f"environment_familiarity: {item.get('environment_familiarity', '')}")
            lines.append(f"object_familiarity: {item.get('object_familiarity', '')}")
            lines.append(f"social_context: {item.get('social_context', '')}")
            lines.append(f"social_detail: {item.get('social_detail', '')}")
            lines.append(f"activity_type: {item.get('activity_type', '')}")
            lines.append(f"rationale: {item.get('rationale', '')}")
            lines.append("")
        else:
            lines.append(f"{index}. {item}")
            lines.append("")
    return lines


def _render_tasks(tasks_payload: dict[str, Any]) -> list[str]:
    tasks = tasks_payload.get("tasks", [])
    if not isinstance(tasks, list) or not tasks:
        return ["No tasks generated."]

    lines: list[str] = []
    for index, item in enumerate(tasks, start=1):
        if isinstance(item, dict):
            lines.append(f"{index}.")
            lines.append(f"task_id: {item.get('task_id', '')}")
            lines.append(f"task_type: {item.get('task_type', item.get('task', ''))}")
            lines.append(f"description: {item.get('description', '')}")
            lines.append(f"confidence: {item.get('confidence', '')}")
            lines.append(f"rationale: {item.get('rationale', '')}")
            lines.append("")
        else:
            lines.append(f"{index}. {item}")
            lines.append("")
    return lines


def _render_question_summary(question_payload: dict[str, Any]) -> list[str]:
    candidates = question_payload.get("candidates", [])

    lines: list[str] = []
    if isinstance(candidates, list) and candidates:
        lines.append("Top candidate details:")
        top = candidates[0]
        if isinstance(top, dict):
            lines.append(f"rank: {top.get('rank', '')}")
            lines.append(f"response: {top.get('response', top.get('question', top.get('text', '')))}")
            lines.append(f"associated_task_id: {top.get('associated_task_id', '')}")
            lines.append(f"associated_context_id: {top.get('associated_context_id', '')}")
            lines.append(f"plausibility_score: {top.get('plausibility_score', top.get('confidence', ''))}")
            lines.append(f"rationale: {top.get('rationale', '')}")
    else:
        lines.append("No question details available.")

    return lines


def _render_question_details(question_payload: dict[str, Any]) -> list[str]:
    ranked_questions = question_payload.get("ranked_questions", [])
    candidates = question_payload.get("candidates", [])
    if not isinstance(ranked_questions, list) or not ranked_questions:
        return ["No ranked questions found."]

    if not isinstance(candidates, list):
        candidates = []

    sorted_candidates = _build_candidate_lookup([item for item in candidates if isinstance(item, dict)])
    lines: list[str] = []
    for index, question in enumerate(ranked_questions, start=1):
        candidate = sorted_candidates[index - 1] if index - 1 < len(sorted_candidates) else {}
        lines.append(f"{index}.")
        lines.append(f"question: {question}")
        lines.append(f"task_type: {candidate.get('task_type', '')}")
        lines.append(f"response_target: {candidate.get('response_target', '')}")
        lines.append("")
    return lines


def _render_match_scores(comparison_payload: dict[str, Any]) -> list[str]:
    score_rows = comparison_payload.get("per_question_scores", [])
    if not isinstance(score_rows, list) or not score_rows:
        return ["No comparison scores available."]

    lines: list[str] = []
    for row in score_rows:
        if not isinstance(row, dict):
            continue
        lines.append(str(row.get("rank", "")) + ".")
        lines.append(f"question: {row.get('generated_question', '')}")
        lines.append(f"generated_task_type: {row.get('generated_task_type', '')}")
        lines.append(f"generated_response_target: {row.get('generated_response_target', '')}")
        lines.append(f"task_type_match_score: {row.get('task_type_match_score', 0)}")
        lines.append(f"target_match_score: {row.get('target_match_score', 0)}")
        lines.append("")
    return lines


def build_readable_output(payload: dict[str, Any]) -> str:
    """Build human-readable report for qualitative analysis."""

    input_data = payload.get("input", {})
    contexts = payload.get("context_agent", {})
    tasks = payload.get("task_agent", {})
    question_agent = payload.get("question_agent", {})
    response_target_agent = payload.get("response_target_agent", {})
    ranked = payload.get("final_ranked_questions", [])
    comparison = payload.get("comparison", {})

    lines: list[str] = []
    lines.append("=== INPUT ===")
    lines.append(f"Image(s): {input_data.get('image_paths', [])}")
    lines.append(f"Description: {input_data.get('description', '')}")
    lines.append(f"Question Category: {input_data.get('question_category', '')}")
    lines.append(f"Ground Truth Response Target: {response_target_agent.get('response_target', '')}")
    lines.append(f"Ground Truth Task Type: {response_target_agent.get('task_type', '')}")
    lines.append("")

    lines.append("=== CONTEXTS ===")
    lines.extend(_render_contexts(contexts))
    lines.append("")

    lines.append("=== TASKS ===")
    lines.extend(_render_tasks(tasks))
    lines.append("")

    lines.append("=== GENERATED QUESTIONS (RANKED) ===")
    if ranked:
        for index, question in enumerate(ranked, start=1):
            lines.append(f"{index}. {question}")
    else:
        lines.append("No ranked questions found.")
    lines.append("")

    lines.append("=== GENERATED QUESTION DETAILS ===")
    lines.extend(_render_question_details(question_agent))
    lines.append("")

    lines.append("=== QUESTION SUMMARY ===")
    lines.extend(_render_question_summary(question_agent))
    lines.append("")

    lines.append("=== GROUND TRUTH ===")
    lines.append(str(payload.get("ground_truth_question", "")))
    lines.append("")

    lines.append("=== COMPARISON ===")
    lines.extend(_render_match_scores(comparison))
    lines.append("Raw comparison JSON:")
    lines.append(json.dumps(comparison, indent=2, ensure_ascii=True))

    return "\n".join(lines)


def save_readable_output(path: Path, text: str) -> None:
    """Write readable text report to disk."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
