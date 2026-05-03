"""Flexible participant interaction loader for diary/inlab BLV VQA data."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from config import PARTICIPANT_DATA_DIR


class DataLoaderError(RuntimeError):
    """Raised when participant data cannot be resolved or parsed."""


def _find_json_file(participant_id: str, source: str) -> Path:
    source = source.lower()

    candidates: list[Path] = []
    if source == "diary":
        candidates.extend(
            [
                PARTICIPANT_DATA_DIR / participant_id / "diary_data" / f"{participant_id}.json",
                PARTICIPANT_DATA_DIR / "diary_data" / f"{participant_id}.json",
                PARTICIPANT_DATA_DIR / participant_id / f"{participant_id}.json",
                PARTICIPANT_DATA_DIR / f"{participant_id}.json",
            ]
        )
    elif source == "inlab":
        candidates.extend(
            [
                PARTICIPANT_DATA_DIR / participant_id / "inlab_data" / f"{participant_id}_inlab.json",
                PARTICIPANT_DATA_DIR / participant_id / "inlab_data" / f"{participant_id}.json",
                PARTICIPANT_DATA_DIR / "inlab_data" / f"{participant_id}_inlab.json",
                PARTICIPANT_DATA_DIR / "inlab_data" / f"{participant_id}.json",
                PARTICIPANT_DATA_DIR / participant_id / f"{participant_id}_inlab.json",
            ]
        )
    else:
        raise DataLoaderError("source must be either 'diary' or 'inlab'.")

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise DataLoaderError(
        f"No JSON file found for participant={participant_id}, source={source}. "
        f"Checked: {[str(c) for c in candidates]}"
    )


def _normalize_interactions(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, list):
        return [item for item in raw if isinstance(item, dict)]

    if isinstance(raw, dict):
        for key in ["interactions", "data", "entries", "records", "items"]:
            maybe = raw.get(key)
            if isinstance(maybe, list):
                return [item for item in maybe if isinstance(item, dict)]

        # Fallback: treat entire dict as a single interaction.
        return [raw]

    raise DataLoaderError("Unsupported JSON schema: expected dict or list at top level.")


def _extract_first_value(payload: dict[str, Any], keys: list[str]) -> Any:
    lowered_map = {str(key).lower(): value for key, value in payload.items()}
    for key in keys:
        if key.lower() in lowered_map:
            return lowered_map[key.lower()]
    return None


def _strip_known_prefix(value: str | None) -> str:
    text = (value or "").strip()
    if not text:
        return ""

    # Common prefixes in BLV transcript data.
    for prefix in ["User:", "Be My AI:", "BeMyAI:", "AI:"]:
        if text.lower().startswith(prefix.lower()):
            return text[len(prefix) :].strip()
    return text


def _extract_from_turns(interaction: dict[str, Any]) -> dict[str, Any]:
    turns_raw = interaction.get("turns", [])
    if not isinstance(turns_raw, list):
        turns_raw = []

    turns: list[dict[str, Any]] = [item for item in turns_raw if isinstance(item, dict)]
    turns.sort(key=lambda t: int(t.get("turn", 10**9)) if str(t.get("turn", "")).isdigit() else 10**9)

    image_paths: list[str] = []
    description = ""
    first_user_response = ""
    all_user_responses: list[str] = []

    for turn in turns:
        local_image_path = turn.get("local_image_path")
        if isinstance(local_image_path, str) and local_image_path.strip():
            image_paths.append(local_image_path.strip())

        if not description:
            ai_text = turn.get("text_ai")
            if isinstance(ai_text, str) and ai_text.strip():
                description = _strip_known_prefix(ai_text)

        if not first_user_response:
            usr_text = turn.get("text_usr")
            if isinstance(usr_text, str) and usr_text.strip():
                first_user_response = _strip_known_prefix(usr_text)

        usr_text = turn.get("text_usr")
        if isinstance(usr_text, str) and usr_text.strip():
            all_user_responses.append(_strip_known_prefix(usr_text))

    annotations = interaction.get("annotations", {})
    question_category = ""
    if isinstance(annotations, dict):
        maybe_qc = annotations.get("question_category")
        if isinstance(maybe_qc, str):
            question_category = maybe_qc

    return {
        "image_paths": image_paths,
        "description": description,
        "ground_truth_question": first_user_response,
        "all_user_responses": all_user_responses,
        "question_category": question_category,
        "annotation_task_type": question_category,
    }


def _find_interaction_keyed_object(raw: dict[str, Any], interaction_id: int) -> tuple[str, dict[str, Any]]:
    direct_key = f"interaction_{interaction_id}"
    if direct_key in raw and isinstance(raw[direct_key], dict):
        return direct_key, raw[direct_key]

    interaction_keys = [
        key
        for key, value in raw.items()
        if isinstance(key, str) and isinstance(value, dict) and re.fullmatch(r"interaction_\d+", key)
    ]
    if interaction_keys:
        raise DataLoaderError(
            f"Could not find key {direct_key} in JSON. "
            f"Available interaction keys: {sorted(interaction_keys)}"
        )

    raise DataLoaderError("No interaction_* keyed objects found in JSON.")


def _coerce_image_paths(value: Any, json_file: Path, participant_id: str, source: str) -> list[str]:
    paths: list[str] = []

    if value is None:
        return []

    if isinstance(value, str):
        paths = [value]
    elif isinstance(value, list):
        paths = [str(item) for item in value if isinstance(item, (str, Path))]
    elif isinstance(value, dict):
        for key in ["paths", "images", "items", "files"]:
            candidate = value.get(key)
            if isinstance(candidate, list):
                paths = [str(item) for item in candidate if isinstance(item, (str, Path))]
                break

    resolved: list[str] = []

    # Candidate image directories by source.
    source_image_dir = (
        PARTICIPANT_DATA_DIR / "diary_data" / f"{participant_id}_images"
        if source == "diary"
        else PARTICIPANT_DATA_DIR / "inlab_data" / f"{participant_id}_inlab_images"
    )

    for image_path in paths:
        candidate = Path(image_path)
        if candidate.is_absolute() and candidate.exists():
            resolved.append(str(candidate))
            continue

        # Try path relative to JSON file first.
        json_relative = (json_file.parent / candidate).resolve()
        if json_relative.exists():
            resolved.append(str(json_relative))
            continue

        # Try source-level image directory.
        source_relative = (source_image_dir / candidate.name).resolve()
        if source_relative.exists():
            resolved.append(str(source_relative))
            continue

        # Keep best-effort unresolved absolute candidate for downstream logging.
        resolved.append(str((json_file.parent / candidate).resolve()))

    return resolved


def load_interaction(participant_id: str, source: str, interaction_id: int) -> dict[str, Any]:
    """Load a single interaction in a schema-tolerant way.

    Args:
        participant_id: Participant identifier, e.g. "P3".
        source: Data source, either "diary" or "inlab".
        interaction_id: 1-based interaction index.

    Returns:
        Normalized interaction payload with keys required by the pipeline.
    """

    if interaction_id < 1:
        raise DataLoaderError("interaction must be >= 1.")

    json_file = _find_json_file(participant_id, source)
    with json_file.open("r", encoding="utf-8") as file:
        raw = json.load(file)

    # Preferred schema: top-level interaction_N objects with annotations + turns.
    if isinstance(raw, dict):
        try:
            interaction_key, interaction = _find_interaction_keyed_object(raw, interaction_id)
            extracted = _extract_from_turns(interaction)
            image_paths = _coerce_image_paths(
                extracted.get("image_paths", []),
                json_file,
                participant_id,
                source.lower(),
            )[:1]

            return {
                "participant_id": participant_id,
                "source": source.lower(),
                "interaction_id": interaction_id,
                "interaction_key": interaction_key,
                "json_file": str(json_file),
                "image_paths": image_paths,
                "description": str(extracted.get("description", "") or ""),
                "ground_truth_question": str(extracted.get("ground_truth_question", "") or ""),
                "all_user_responses": [
                    str(item)
                    for item in extracted.get("all_user_responses", [])
                    if str(item).strip()
                ],
                "question_category": str(extracted.get("question_category", "") or ""),
                "annotation_task_type": str(extracted.get("annotation_task_type", "") or ""),
                "raw_interaction": interaction,
            }
        except DataLoaderError as exc:
            # Fall back to older schemas below.
            if "No interaction_* keyed objects found" not in str(exc):
                raise

    interactions = _normalize_interactions(raw)

    if interaction_id > len(interactions):
        raise DataLoaderError(
            f"interaction={interaction_id} is out of range for {json_file.name}; "
            f"found {len(interactions)} interaction(s)."
        )

    interaction = interactions[interaction_id - 1]

    image_raw = _extract_first_value(
        interaction,
        [
            "image_paths",
            "images",
            "image_path",
            "image",
            "photo_paths",
            "photo",
        ],
    )
    description = _extract_first_value(
        interaction,
        [
            "ai_generated_description",
            "bemyai_output",
            "be_my_ai_output",
            "description",
            "ai_description",
        ],
    )
    ground_truth_question = _extract_first_value(
        interaction,
        [
            "ground_truth_question",
            "user_question",
            "question",
            "participant_question",
            "gt_question",
        ],
    )

    image_paths = _coerce_image_paths(image_raw, json_file, participant_id, source.lower())[:1]

    return {
        "participant_id": participant_id,
        "source": source.lower(),
        "interaction_id": interaction_id,
        "json_file": str(json_file),
        "image_paths": image_paths,
        "description": str(description or ""),
        "ground_truth_question": str(ground_truth_question or ""),
        "all_user_responses": [str(ground_truth_question or "")] if str(ground_truth_question or "").strip() else [],
        "question_category": "",
        "annotation_task_type": "",
        "raw_interaction": interaction,
    }


def load_all_interactions(participant_id: str, source: str) -> list[dict[str, Any]]:
    """Load all interactions for a participant/source in normalized pipeline schema."""

    source = source.lower().strip()
    if source not in {"diary", "inlab"}:
        raise DataLoaderError("source must be either 'diary' or 'inlab'.")

    json_file = _find_json_file(participant_id, source)
    with json_file.open("r", encoding="utf-8") as file:
        raw = json.load(file)

    payloads: list[dict[str, Any]] = []

    if isinstance(raw, dict):
        keyed_items = [
            (key, value)
            for key, value in raw.items()
            if isinstance(key, str) and isinstance(value, dict) and re.fullmatch(r"interaction_\d+", key)
        ]
        if keyed_items:
            keyed_items.sort(key=lambda kv: int(kv[0].split("_")[-1]))
            for key, interaction in keyed_items:
                interaction_id = int(key.split("_")[-1])
                extracted = _extract_from_turns(interaction)
                image_paths = _coerce_image_paths(
                    extracted.get("image_paths", []),
                    json_file,
                    participant_id,
                    source,
                )[:1]
                payloads.append(
                    {
                        "participant_id": participant_id,
                        "source": source,
                        "interaction_id": interaction_id,
                        "interaction_key": key,
                        "json_file": str(json_file),
                        "image_paths": image_paths,
                        "description": str(extracted.get("description", "") or ""),
                        "ground_truth_question": str(extracted.get("ground_truth_question", "") or ""),
                        "all_user_responses": [
                            str(item)
                            for item in extracted.get("all_user_responses", [])
                            if str(item).strip()
                        ],
                        "question_category": str(extracted.get("question_category", "") or ""),
                        "annotation_task_type": str(extracted.get("annotation_task_type", "") or ""),
                        "raw_interaction": interaction,
                    }
                )
            return payloads

    interactions = _normalize_interactions(raw)
    for index, interaction in enumerate(interactions, start=1):
        image_raw = _extract_first_value(
            interaction,
            [
                "image_paths",
                "images",
                "image_path",
                "image",
                "photo_paths",
                "photo",
            ],
        )
        description = _extract_first_value(
            interaction,
            [
                "ai_generated_description",
                "bemyai_output",
                "be_my_ai_output",
                "description",
                "ai_description",
            ],
        )
        ground_truth_question = _extract_first_value(
            interaction,
            [
                "ground_truth_question",
                "user_question",
                "question",
                "participant_question",
                "gt_question",
            ],
        )
        image_paths = _coerce_image_paths(image_raw, json_file, participant_id, source)[:1]

        payloads.append(
            {
                "participant_id": participant_id,
                "source": source,
                "interaction_id": index,
                "json_file": str(json_file),
                "image_paths": image_paths,
                "description": str(description or ""),
                "ground_truth_question": str(ground_truth_question or ""),
                "all_user_responses": [str(ground_truth_question or "")] if str(ground_truth_question or "").strip() else [],
                "question_category": "",
                "annotation_task_type": "",
                "raw_interaction": interaction,
            }
        )

    return payloads
