"""Task generation agent."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from config import PROMPTS_DIR
from utils.response_parser import extract_json_value, normalize_task_output
from utils.openai_client import call_model


def _read_prompt(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8").strip()


class TaskAgent:
    """Generate structured task hypotheses for one interaction."""

    def __init__(self) -> None:
        shared = _read_prompt(PROMPTS_DIR / "shared_prompt.txt")
        specific = _read_prompt(PROMPTS_DIR / "task_prompt.txt")
        self.prompt_template = f"{shared}\n\n{specific}".strip()

    def run(
        self,
        image_paths: list[str],
        question_category: str = "",
        optional_prompt: str = "",
    ) -> dict[str, Any]:
        prompt_prefix = self.prompt_template
        if optional_prompt.strip():
            prompt_prefix = f"{optional_prompt.strip()}\n\n{prompt_prefix}"

        prompt = (
            f"{prompt_prefix}\n\n"
            "INPUT:\n"
            f"question_category: {question_category}\n"
            f"image_paths: {image_paths}\n\n"
            "Return valid JSON only. Recommended format:\n"
            "{\"tasks\": [{\"task\": \"...\", \"confidence\": 0.0, \"evidence\": \"...\"}]}"
        )

        response_text = call_model(prompt=prompt, image_paths=image_paths)
        parsed = extract_json_value(response_text)
        return normalize_task_output(parsed if parsed is not None else response_text)
