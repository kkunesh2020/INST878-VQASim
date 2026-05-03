"""Whole-interaction response target inference agent."""

from __future__ import annotations

import json
from pathlib import Path

from config import PROMPTS_DIR
from utils.openai_client import call_model
from utils.response_parser import extract_json_value, normalize_response_target_output
from agents.response_target_agent import ResponseTargetAgent


def _read_prompt(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8").strip()


class InteractionTargetAgent:
    """Infer one overall target from all participant responses in an interaction."""

    def __init__(self) -> None:
        shared = _read_prompt(PROMPTS_DIR / "shared_prompt.txt")
        specific = _read_prompt(PROMPTS_DIR / "interaction_target_prompt.txt")
        self.prompt_template = f"{shared}\n\n{specific}".strip()

    def run(
        self,
        image_paths: list[str],
        all_user_responses: list[str],
        question_category: str = "",
        optional_prompt: str = "",
    ) -> dict[str, object]:
        prompt_prefix = self.prompt_template
        if optional_prompt.strip():
            prompt_prefix = f"{optional_prompt.strip()}\n\n{prompt_prefix}"

        prompt = (
            f"{prompt_prefix}\n\n"
            "INPUT:\n"
            f"all_user_responses: {json.dumps(all_user_responses, ensure_ascii=True)}\n"
            f"question_category: {question_category}\n\n"
            "Return valid JSON only. Recommended format:\n"
            "{\"turn_targets\": [{\"response_target\": \"...\", \"task_type\": \"...\", \"confidence\": \"high|medium|low\", \"rationale\": \"...\"}]}"
        )

        response_text = call_model(prompt=prompt, image_paths=image_paths)
        parsed = extract_json_value(response_text)

        turn_targets: list[dict[str, object]] = []
        # If the model returned a list of targets, normalize each
        if isinstance(parsed, list):
            for item in parsed:
                norm = normalize_response_target_output(item if item is not None else {})
                # strip raw_output from per-turn entry to reduce redundancy
                turn_targets.append(
                    {
                        "response_target": norm.get("response_target", ""),
                        "task_type": norm.get("task_type", ""),
                        "confidence": norm.get("confidence", ""),
                        "rationale": norm.get("rationale", ""),
                    }
                )
        elif isinstance(parsed, dict):
            # Preferred container key: turn_targets
            container = parsed.get("turn_targets") if isinstance(parsed.get("turn_targets"), list) else None
            if container is not None:
                for item in container:
                    norm = normalize_response_target_output(item if item is not None else {})
                    turn_targets.append(
                        {
                            "response_target": norm.get("response_target", ""),
                            "task_type": norm.get("task_type", ""),
                            "confidence": norm.get("confidence", ""),
                            "rationale": norm.get("rationale", ""),
                        }
                    )
            else:
                # Try to interpret top-level dict as a single turn target (apply to first turn)
                single = normalize_response_target_output(parsed)
                if all_user_responses:
                    turn_targets.append(
                        {
                            "response_target": single.get("response_target", ""),
                            "task_type": single.get("task_type", ""),
                            "confidence": single.get("confidence", ""),
                            "rationale": single.get("rationale", ""),
                        }
                    )
                    # Fill remaining turns with empty placeholders
                    for _ in range(len(all_user_responses) - 1):
                        turn_targets.append({"response_target": "", "task_type": question_category or "", "confidence": "", "rationale": ""})
                else:
                    # No user responses available; return the single normalized target as one-item list
                    turn_targets.append(
                        {
                            "response_target": single.get("response_target", ""),
                            "task_type": single.get("task_type", ""),
                            "confidence": single.get("confidence", ""),
                            "rationale": single.get("rationale", ""),
                        }
                    )
        else:
            # Fallback: no parsed JSON; create one target per user response as empty or fallback
            if all_user_responses:
                for _ in all_user_responses:
                    turn_targets.append({"response_target": "", "task_type": question_category or "", "confidence": "", "rationale": ""})

        # If the model did not return one target per user response, fall back
        # to extracting per-turn targets using the single-response agent so
        # targets match the style used elsewhere in the pipeline.
        if all_user_responses and len(turn_targets) != len(all_user_responses):
            fallback: list[dict[str, object]] = []
            rta = ResponseTargetAgent()
            for usr in all_user_responses:
                try:
                    single = rta.run(image_paths, "", str(usr), question_category=question_category, optional_prompt=optional_prompt)
                except Exception:
                    single = {"response_target": "", "task_type": question_category or "", "confidence": "", "rationale": ""}
                fallback.append(
                    {
                        "response_target": str(single.get("response_target", "") or ""),
                        "task_type": str(single.get("task_type", "") or question_category or ""),
                        "confidence": str(single.get("confidence", "") or ""),
                        "rationale": str(single.get("rationale", "") or ""),
                    }
                )
            turn_targets = fallback

        # Enforce question_category if provided for each turn
        if question_category.strip():
            for item in turn_targets:
                item["task_type"] = question_category.strip()

        return {"turn_targets": turn_targets, "raw_output": parsed if parsed is not None else response_text}
