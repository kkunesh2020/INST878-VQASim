"""Batch runner for BLV VQA multi-agent pipeline.

Runs all valid interactions for a participant/source while skipping records
without participant questions (for example, question_type/category of
"No question").
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import Any

from main import run as run_single_interaction
from config import OUTPUT_JSON_DIR
from utils.data_loader import DataLoaderError, load_all_interactions
from utils.formatter import save_json_output

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

_NO_QUESTION_PATTERN = re.compile(r"\bno\s*question\b", re.IGNORECASE)


def _is_no_question_category(question_category: str) -> bool:
    return bool(_NO_QUESTION_PATTERN.search((question_category or "").strip()))


def _has_participant_response(interaction: dict[str, Any]) -> bool:
    question_category = str(interaction.get("question_category", "") or "")
    if _is_no_question_category(question_category):
        return False

    ground_truth_question = str(interaction.get("ground_truth_question", "") or "").strip()
    return bool(ground_truth_question)


def _build_batch_interaction_summary(interaction_id: int, output_payload: dict[str, Any]) -> dict[str, Any]:
    comparison = output_payload.get("comparison", {})
    per_question_scores = comparison.get("per_question_scores", []) if isinstance(comparison, dict) else []
    generated_responses: list[dict[str, Any]] = []

    if isinstance(per_question_scores, list):
        for row in per_question_scores:
            if not isinstance(row, dict):
                continue
            if row.get("rank") == "ground_truth":
                continue
            generated_responses.append(
                {
                    "rank": row.get("rank", ""),
                    "question": row.get("generated_question", ""),
                    "response_target": row.get("generated_response_target", ""),
                    "task_type": row.get("generated_task_type", ""),
                    "target_match_score": row.get("target_match_score", 0),
                    "matched_user_turn": row.get("matched_user_turn", None),
                    "matched_user_turn_target": row.get("matched_user_turn_target", ""),
                    "matched_user_turn_score": row.get("matched_user_turn_score", 0),
                    "task_type_match_score": row.get("task_type_match_score", 0),
                }
            )

    return {
        "interaction": interaction_id,
        "status": "success",
        "ground_truth": {
            "question": output_payload.get("ground_truth_question", ""),
            "response_target": comparison.get("ground_truth_response_target", "") if isinstance(comparison, dict) else "",
            "turn_targets": comparison.get("turn_targets", []) if isinstance(comparison, dict) else [],
            "task_type": comparison.get("ground_truth_task_type", "") if isinstance(comparison, dict) else "",
        },
        "generated_responses": generated_responses,
    }


def run_batch(
    participant: str,
    source: str,
    include_demographics: bool = False,
    include_fewshot: bool = False,
    dry_run: bool = False,
    stop_on_error: bool = False,
) -> None:
    source = source.lower().strip()
    if source not in {"diary", "inlab"}:
        raise ValueError("--source must be one of: diary, inlab")

    try:
        interactions = load_all_interactions(participant, source)
    except (DataLoaderError, FileNotFoundError, ValueError) as exc:
        logger.error("Failed to load interactions: %s", exc)
        raise SystemExit(1)

    valid_interactions = [interaction for interaction in interactions if _has_participant_response(interaction)]

    if not valid_interactions:
        logger.warning(
            "No valid interactions found for participant=%s source=%s (after filtering).",
            participant,
            source,
        )
        return

    skipped_count = len(interactions) - len(valid_interactions)
    interaction_ids = [int(item.get("interaction_id", -1)) for item in valid_interactions]

    logger.info(
        "Batch selection complete for participant=%s source=%s: %s selected, %s skipped.",
        participant,
        source,
        len(valid_interactions),
        skipped_count,
    )
    logger.info("Selected interaction IDs: %s", interaction_ids)

    if dry_run:
        logger.info("Dry run enabled. No pipeline runs were executed.")
        return

    success_count = 0
    failed_ids: list[int] = []
    batch_results: list[dict[str, Any]] = []

    for interaction in valid_interactions:
        interaction_id = int(interaction["interaction_id"])
        logger.info(
            "Running interaction %s for participant=%s source=%s",
            interaction_id,
            participant,
            source,
        )

        try:
            output_payload = run_single_interaction(
                participant=participant,
                source=source,
                interaction=interaction_id,
                include_demographics=include_demographics,
                include_fewshot=include_fewshot,
            )
            success_count += 1
            batch_results.append(_build_batch_interaction_summary(interaction_id, output_payload))
        except SystemExit as exc:
            failed_ids.append(interaction_id)
            logger.error("Interaction %s failed with exit code: %s", interaction_id, exc.code)
            batch_results.append(
                {
                    "interaction": interaction_id,
                    "status": "failed",
                    "ground_truth": {
                        "question": "",
                        "response_target": "",
                        "task_type": "",
                    },
                    "generated_responses": [],
                    "error": f"SystemExit({exc.code})",
                }
            )
            if stop_on_error:
                break
        except Exception:
            failed_ids.append(interaction_id)
            logger.exception("Interaction %s failed with an unexpected error", interaction_id)
            batch_results.append(
                {
                    "interaction": interaction_id,
                    "status": "failed",
                    "ground_truth": {
                        "question": "",
                        "response_target": "",
                        "task_type": "",
                    },
                    "generated_responses": [],
                    "error": "unexpected error",
                }
            )
            if stop_on_error:
                break

    logger.info(
        "Batch finished for participant=%s source=%s | success=%s | failed=%s",
        participant,
        source,
        success_count,
        len(failed_ids),
    )
    if failed_ids:
        logger.info("Failed interaction IDs: %s", failed_ids)

    batch_suffix = ""
    if include_demographics:
        batch_suffix += "_demo"
    if include_fewshot:
        batch_suffix += "_fewshot"

    batch_output_path: Path = OUTPUT_JSON_DIR / f"{participant}_{source}_batch_comparison{batch_suffix}.json"
    batch_output_payload: dict[str, Any] = {
        "participant": participant,
        "source": source,
        "include_demographics": include_demographics,
        "include_fewshot": include_fewshot,
        "selected_interaction_ids": interaction_ids,
        "skipped_interaction_count": skipped_count,
        "success_count": success_count,
        "failed_interaction_ids": failed_ids,
        "comparison_results": batch_results,
    }
    save_json_output(batch_output_path, batch_output_payload)
    logger.info("Saved batch comparison output: %s", batch_output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Run the BLV VQA pipeline in batch mode for all valid interactions "
            "for a participant/source."
        )
    )
    parser.add_argument("--participant", required=True, help="Participant ID, e.g. P3")
    parser.add_argument("--source", required=True, choices=["diary", "inlab"], help="Data source")
    parser.add_argument(
        "--include-demographics",
        action="store_true",
        help="Include participant demographics prompt context.",
    )
    parser.add_argument(
        "--include-fewshot",
        action="store_true",
        help="Include random participant few-shot interaction examples.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only list selected interaction IDs; do not run the pipeline.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop the batch at the first failed interaction.",
    )
    args = parser.parse_args()

    run_batch(
        participant=args.participant,
        source=args.source,
        include_demographics=args.include_demographics,
        include_fewshot=args.include_fewshot,
        dry_run=args.dry_run,
        stop_on_error=args.stop_on_error,
    )
