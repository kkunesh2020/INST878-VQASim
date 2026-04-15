"""Ground-truth response target inference agent."""

from __future__ import annotations

import json
from pathlib import Path

from config import PROMPTS_DIR
from utils.openai_client import call_model
from utils.response_parser import extract_json_value, normalize_response_target_output


def _read_prompt(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8").strip()


class ResponseTargetAgent:
    """Infer response target and task type from participant's ground-truth response."""

    def __init__(self) -> None:
        shared = _read_prompt(PROMPTS_DIR / "shared_prompt.txt")
        specific = _read_prompt(PROMPTS_DIR / "response_target_prompt.txt")
        self.prompt_template = f"{shared}\n\n{specific}".strip()

    def run(
        self,
        image_paths: list[str],
        description: str,
        ground_truth_question: str,
        question_category: str = "",
        optional_prompt: str = "",
    ) -> dict[str, object]:
        prompt_prefix = self.prompt_template
        if optional_prompt.strip():
            prompt_prefix = f"{optional_prompt.strip()}\n\n{prompt_prefix}"

        prompt = (
            f"{prompt_prefix}\n\n"
            "INPUT:\n"
            f"description: {description}\n"
            f"question_category: {question_category}\n"
            f"ground_truth_question: {ground_truth_question}\n"
            f"image_paths: {image_paths}\n\n"
            "Return valid JSON only. Required format:\n"
            "{"
            '"response_target": "...", '
            '"task_type": "Reading", '
            '"confidence": "high|medium|low", '
            '"rationale": "..."'
            "}"
        )

        response_text = call_model(prompt=prompt, image_paths=image_paths)
        parsed = extract_json_value(response_text)
        return normalize_response_target_output(parsed if parsed is not None else response_text)
