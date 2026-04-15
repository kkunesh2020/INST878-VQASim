"""Helpers for parsing and normalizing model JSON outputs."""

from __future__ import annotations

import json
from typing import Any


def extract_json_value(text: str) -> Any:
    """Extract JSON from model output.

    Supports:
    - pure JSON
    - fenced JSON blocks
    - prose with an embedded JSON object/list
    """

    cleaned = (text or "").strip()
    if not cleaned:
        return None

    # Remove common markdown fences.
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Try extracting the widest JSON object/list from within surrounding prose.
    candidates = []
    for start_char, end_char in [("{", "}"), ("[", "]")]:
        start = cleaned.find(start_char)
        end = cleaned.rfind(end_char)
        if start != -1 and end != -1 and end > start:
            candidates.append(cleaned[start : end + 1])

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue

    return None


def normalize_context_output(model_output: Any) -> dict[str, Any]:
    """Normalize context-agent output to a consistent dictionary shape."""

    if isinstance(model_output, dict):
        if isinstance(model_output.get("contexts"), list):
            return {"contexts": model_output["contexts"], "raw_output": model_output}
        return {"contexts": [model_output], "raw_output": model_output}

    if isinstance(model_output, list):
        return {"contexts": model_output, "raw_output": model_output}

    return {"contexts": [], "raw_output": model_output}


def normalize_task_output(model_output: Any) -> dict[str, Any]:
    """Normalize task-agent output to a consistent dictionary shape."""

    if isinstance(model_output, dict):
        if isinstance(model_output.get("tasks"), list):
            return {"tasks": model_output["tasks"], "raw_output": model_output}
        return {"tasks": [model_output], "raw_output": model_output}

    if isinstance(model_output, list):
        return {"tasks": model_output, "raw_output": model_output}

    return {"tasks": [], "raw_output": model_output}


def normalize_question_output(model_output: Any) -> dict[str, Any]:
    """Normalize question-agent output and derive ranked question strings."""

    ranked_questions: list[str] = []
    items: list[dict[str, Any]] = []

    if isinstance(model_output, dict):
        if isinstance(model_output.get("ranked_questions"), list):
            ranked_questions = [str(item) for item in model_output["ranked_questions"]]
        elif isinstance(model_output.get("questions"), list):
            ranked_questions = [str(item) for item in model_output["questions"]]

        if isinstance(model_output.get("candidates"), list):
            items = [item for item in model_output["candidates"] if isinstance(item, dict)]
        elif isinstance(model_output.get("responses"), list):
            items = [item for item in model_output["responses"] if isinstance(item, dict)]
        else:
            items = [model_output]

    elif isinstance(model_output, list):
        items = [item for item in model_output if isinstance(item, dict)]
    else:
        items = []

    # Some model outputs wrap the actual ranked items in a single container dict:
    # {"candidates": [{"ranked_questions": [{...}, ...]}]}
    if items:
        flattened: list[dict[str, Any]] = []
        for item in items:
            nested_ranked = item.get("ranked_questions") if isinstance(item, dict) else None
            nested_candidates = item.get("candidates") if isinstance(item, dict) else None

            if isinstance(nested_ranked, list) and nested_ranked:
                flattened.extend([sub for sub in nested_ranked if isinstance(sub, dict)])
                continue

            if isinstance(nested_candidates, list) and nested_candidates:
                flattened.extend([sub for sub in nested_candidates if isinstance(sub, dict)])
                continue

            flattened.append(item)

        items = flattened

    if not ranked_questions and items:
        sortable = []
        for index, item in enumerate(items):
            rank = item.get("rank")
            try:
                sort_rank = int(rank)
            except Exception:
                sort_rank = index + 1
            response = item.get("response") or item.get("question") or item.get("text") or ""
            sortable.append((sort_rank, str(response), item))
        sortable.sort(key=lambda x: x[0])
        ranked_questions = [response for _, response, _ in sortable if response]
        items = [item for _, _, item in sortable]

    return {
        "ranked_questions": ranked_questions,
        "candidates": items,
        "raw_output": model_output,
    }


def normalize_response_target_output(model_output: Any) -> dict[str, Any]:
    """Normalize response-target output to a consistent dictionary shape."""

    if isinstance(model_output, dict):
        return {
            "response_target": str(model_output.get("response_target", "") or ""),
            "task_type": str(model_output.get("task_type", "") or ""),
            "confidence": str(model_output.get("confidence", "") or ""),
            "rationale": str(model_output.get("rationale", "") or ""),
            "raw_output": model_output,
        }

    return {
        "response_target": "",
        "task_type": "",
        "confidence": "",
        "rationale": "",
        "raw_output": model_output,
    }
