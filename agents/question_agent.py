"""Follow-up question generation agent."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from config import PROMPTS_DIR
from utils.response_parser import extract_json_value, normalize_question_output
from utils.openai_client import call_model


def _read_prompt(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8").strip()


class QuestionAgent:
    """Generate ranked follow-up questions using prior context and task outputs."""

    def __init__(self) -> None:
        shared = _read_prompt(PROMPTS_DIR / "shared_prompt.txt")
        specific = _read_prompt(PROMPTS_DIR / "question_prompt.txt")
        self.prompt_template = f"{shared}\n\n{specific}".strip()

    def run(
        self,
        image_paths: list[str],
        description: str,
        contexts: dict[str, Any],
        tasks: dict[str, Any],
        question_category: str = "",
        optional_prompt: str = "",
    ) -> dict[str, Any]:
        prompt_prefix = self.prompt_template
        if optional_prompt.strip():
            prompt_prefix = f"{optional_prompt.strip()}\n\n{prompt_prefix}"

        prompt = (
            f"{prompt_prefix}\n\n"
            "INPUT:\n"
            f"description: {description}\n"
            f"question_category: {question_category}\n"
            f"image_paths: {image_paths}\n"
            f"contexts: {json.dumps(contexts, ensure_ascii=True)}\n"
            f"tasks: {json.dumps(tasks, ensure_ascii=True)}\n\n"
            "Return valid JSON only. Recommended format:\n"
            "{\"ranked_questions\": [\"...\", \"...\"], \"rationale\": \"...\"}"
        )

        response_text = call_model(prompt=prompt, image_paths=image_paths)
        parsed = extract_json_value(response_text)
        return normalize_question_output(parsed if parsed is not None else response_text)
