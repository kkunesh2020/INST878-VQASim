"""Helpers to build optional demographics and few-shot prompt blocks."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from config import PARTICIPANT_DATA_DIR, PROMPTS_DIR
from utils.data_loader import DataLoaderError, load_all_interactions


def _read_prompt(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8").strip()


def _load_demographics_json() -> dict[str, Any]:
    demographics_path = PARTICIPANT_DATA_DIR / "demographics" / "demographics.json"
    if not demographics_path.exists():
        raise DataLoaderError(f"Demographics file not found: {demographics_path}")

    with demographics_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if isinstance(payload, dict):
        # Preferred schema: {"P1": {...}, "P2": {...}}
        if all(isinstance(key, str) and isinstance(value, dict) for key, value in payload.items()):
            return payload

        # Backward-compatible schema: {"participants": [{"pid": "P1", ...}]}
        participants = payload.get("participants")
        if isinstance(participants, list):
            normalized: dict[str, dict[str, Any]] = {}
            for item in participants:
                if not isinstance(item, dict):
                    continue
                pid = item.get("pid")
                if isinstance(pid, str) and pid.strip():
                    normalized[pid.strip()] = {k: v for k, v in item.items() if str(k).lower() != "pid"}
            return normalized

    raise DataLoaderError(
        "Unsupported demographics schema. Expected {\"P1\": {...}} or {\"participants\": [...]}"
    )


def build_demographics_block(participant_id: str) -> dict[str, Any]:
    """Build rendered optional demographics prompt block for one participant."""

    demographics_by_pid = _load_demographics_json()
    entry = demographics_by_pid.get(participant_id)
    if not isinstance(entry, dict):
        raise DataLoaderError(f"No demographics entry found for participant={participant_id}")

    template = _read_prompt(PROMPTS_DIR / "optional" / "demographics_prompt.txt")
    formatted = json.dumps(entry, indent=2, ensure_ascii=True)
    rendered = template.replace("{INSERT DEMOGRAPHICS}", formatted)

    return {
        "prompt": rendered,
        "demographics": entry,
    }


def build_fewshot_block(
    participant_id: str,
    source: str,
    sample_size: int = 6,
    seed: int | None = None,
) -> dict[str, Any]:
    """Build rendered optional few-shot prompt block using random interactions."""

    interactions = load_all_interactions(participant_id, source)
    if not interactions:
        raise DataLoaderError(f"No interactions found for participant={participant_id}, source={source}")

    rng = random.Random(seed)
    k = min(sample_size, len(interactions))
    selected = rng.sample(interactions, k=k)

    examples: list[dict[str, Any]] = []
    for item in selected:
        image_paths = item.get("image_paths")
        first_image = ""
        if isinstance(image_paths, list) and image_paths:
            first_image = str(image_paths[0])

        examples.append(
            {
                "interaction_id": item.get("interaction_id"),
                "image": first_image,
                "ai_description": item.get("description", ""),
                "participant_first_response": item.get("ground_truth_question", ""),
                "question_type": item.get("question_category", ""),
            }
        )

    template = _read_prompt(PROMPTS_DIR / "optional" / "fewshot_prompt.txt")
    formatted = json.dumps(examples, indent=2, ensure_ascii=True)
    rendered = template.replace("{INSERT 4\u20136 EXAMPLES}", formatted).replace(
        "{INSERT 4-6 EXAMPLES}", formatted
    )

    return {
        "prompt": rendered,
        "examples": examples,
    }