"""CLI entrypoint for BLV VQA multi-agent analysis pipeline."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

from agents.context_agent import ContextAgent
from agents.interaction_target_agent import InteractionTargetAgent
from agents.question_agent import QuestionAgent
from agents.response_target_agent import ResponseTargetAgent
from agents.task_agent import TaskAgent
from config import OUTPUT_JSON_DIR, OUTPUT_READABLE_DIR
from utils.data_loader import DataLoaderError, load_interaction
from utils.formatter import (
    build_comparison,
    build_readable_output,
    save_json_output,
    save_readable_output,
)
from utils.image_utils import validate_image_paths
from utils.optional_prompt_context import build_demographics_block, build_fewshot_block

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def _extract_ranked_questions(question_output: dict[str, Any]) -> list[str]:
    for key in ["ranked_questions", "questions", "final_questions", "candidates"]:
        value = question_output.get(key)
        if isinstance(value, list):
            return [str(item) for item in value]
    return []


def run(
    participant: str,
    source: str,
    interaction: int,
    include_demographics: bool = False,
    include_fewshot: bool = False,
) -> dict[str, Any]:
    """Run context -> task -> question agent pipeline for one interaction."""

    source = source.lower().strip()
    if source not in {"diary", "inlab"}:
        raise ValueError("--source must be one of: diary, inlab")

    try:
        logger.info("Loading interaction for participant=%s source=%s interaction=%s", participant, source, interaction)
        interaction_payload = load_interaction(participant, source, interaction)
    except (DataLoaderError, FileNotFoundError, ValueError) as exc:
        logger.error("Failed to load interaction: %s", exc)
        raise SystemExit(1)

    image_paths = interaction_payload.get("image_paths", [])[:1]
    valid_images, missing_images = validate_image_paths(image_paths)
    if missing_images:
        logger.warning("Missing image files: %s", missing_images)

    description = interaction_payload.get("description", "")
    ground_truth_question = interaction_payload.get("ground_truth_question", "")
    all_user_responses = [
        str(item)
        for item in interaction_payload.get("all_user_responses", [])
        if str(item).strip()
    ]
    if not all_user_responses and str(ground_truth_question).strip():
        all_user_responses = [str(ground_truth_question)]
    question_category = interaction_payload.get("question_category", "")

    optional_prompt_sections: list[str] = []
    optional_prompt_metadata: dict[str, Any] = {
        "demographics_enabled": include_demographics,
        "fewshot_enabled": include_fewshot,
    }

    if include_demographics:
        logger.info("Loading optional demographics prompt context")
        try:
            demographics_block = build_demographics_block(participant)
            optional_prompt_sections.append(str(demographics_block.get("prompt", "")))
            optional_prompt_metadata["demographics"] = demographics_block.get("demographics", {})
        except (DataLoaderError, FileNotFoundError, ValueError) as exc:
            logger.error("Failed to build demographics prompt context: %s", exc)
            raise SystemExit(1)

    if include_fewshot:
        logger.info("Loading optional few-shot prompt context")
        try:
            fewshot_block = build_fewshot_block(participant, source, sample_size=6)
            optional_prompt_sections.append(str(fewshot_block.get("prompt", "")))
            optional_prompt_metadata["fewshot_examples"] = fewshot_block.get("examples", [])
        except (DataLoaderError, FileNotFoundError, ValueError) as exc:
            logger.error("Failed to build few-shot prompt context: %s", exc)
            raise SystemExit(1)

    optional_prompt = "\n\n".join(section.strip() for section in optional_prompt_sections if section.strip())

    logger.info("Running ContextAgent")
    context_output = ContextAgent().run(
        valid_images,
        question_category=question_category,
        optional_prompt=optional_prompt,
    )

    logger.info("Running TaskAgent")
    task_output = TaskAgent().run(
        valid_images,
        question_category=question_category,
        optional_prompt=optional_prompt,
    )

    logger.info("Running QuestionAgent")
    question_output = QuestionAgent().run(
        valid_images,
        description,
        context_output,
        task_output,
        question_category=question_category,
        optional_prompt=optional_prompt,
    )

    logger.info("Running ResponseTargetAgent")
    response_target_output = ResponseTargetAgent().run(
        valid_images,
        description,
        ground_truth_question,
        question_category=question_category,
        optional_prompt=optional_prompt,
    )
    response_target_output["task_type"] = str(question_category or response_target_output.get("task_type", ""))

    logger.info("Running InteractionTargetAgent (per-turn)")
    interaction_target_output = InteractionTargetAgent().run(
        valid_images,
        all_user_responses,
        question_category=question_category,
        optional_prompt=optional_prompt,
    )

    ranked_questions = _extract_ranked_questions(question_output)
    comparison = build_comparison(
        question_output,
        ground_truth_question,
        response_target_output,
        interaction_target_output,
    )

    output_payload: dict[str, Any] = {
        "input": {
            "participant": participant,
            "source": source,
            "interaction": interaction,
            "json_file": interaction_payload.get("json_file"),
            "image_paths": valid_images,
            "missing_image_paths": missing_images,
            "description": description,
            "all_user_responses": all_user_responses,
            "question_category": question_category,
            "optional_prompt": optional_prompt_metadata,
        },
        "context_agent": context_output,
        "task_agent": task_output,
        "question_agent": question_output,
        "response_target_agent": response_target_output,
        "interaction_target_agent": interaction_target_output,
        "final_ranked_questions": ranked_questions,
        "ground_truth_question": ground_truth_question,
        "comparison": comparison,
        "raw_interaction": interaction_payload.get("raw_interaction", {}),
    }

    suffix_tags: list[str] = []
    if include_demographics:
        suffix_tags.append("demo")
    if include_fewshot:
        suffix_tags.append("fewshot")

    suffix = ""
    if suffix_tags:
        suffix = "_" + "_".join(suffix_tags)

    base_name = f"{participant}_{source}_{interaction}{suffix}"
    json_path: Path = OUTPUT_JSON_DIR / f"{base_name}.json"
    readable_path: Path = OUTPUT_READABLE_DIR / f"{base_name}.txt"

    save_json_output(json_path, output_payload)
    readable_report = build_readable_output(output_payload)
    save_readable_output(readable_path, readable_report)

    logger.info("Saved JSON output: %s", json_path)
    logger.info("Saved readable output: %s", readable_path)

    return output_payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the BLV VQA multi-agent pipeline.")
    parser.add_argument("--participant", required=True, help="Participant ID, e.g. P3")
    parser.add_argument("--source", required=True, choices=["diary", "inlab"], help="Data source")
    parser.add_argument("--interaction", required=True, type=int, help="1-based interaction index")
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
    args = parser.parse_args()
    run(
        args.participant,
        args.source,
        args.interaction,
        include_demographics=args.include_demographics,
        include_fewshot=args.include_fewshot,
    )
