"""Ground-truth response target inference agent."""

from __future__ import annotations

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
            f"ground_truth_question: {ground_truth_question}\n"
            f"question_category: {question_category}\n\n"
            "Focus on the participant's question text and extract only the information target.\n"
            "Keep response_target short, concrete, and not overly specific.\n"
            "Use question_category exactly as provided for task_type.\n\n"
            "Return valid JSON only. Required format:\n"
            "{"
            '"response_target": "...", '
            '"task_type": "...", '
            '"confidence": "high|medium|low", '
            '"rationale": "..."'
            "}"
        )

        response_text = call_model(prompt=prompt, image_paths=image_paths)
        parsed = extract_json_value(response_text)
        normalized = normalize_response_target_output(parsed if parsed is not None else response_text)
        if question_category.strip():
            normalized["task_type"] = question_category.strip()
        return normalized
