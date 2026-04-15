"""Centralized OpenAI client and model-calling helpers."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

from openai import OpenAI

from config import DEFAULT_MODEL, OPENAI_API_KEY


class OpenAIClientError(RuntimeError):
    """Raised when model invocation fails or is misconfigured."""


def _get_client() -> OpenAI:
    if not OPENAI_API_KEY:
        raise OpenAIClientError(
            "OPENAI_API_KEY is not set. Add it to your .env file before running."
        )
    return OpenAI(api_key=OPENAI_API_KEY)


def _encode_image_to_data_url(image_path: str | Path) -> str:
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")

    suffix = path.suffix.lower().lstrip(".") or "jpeg"
    mime_type = "image/jpeg" if suffix in {"jpg", "jpeg"} else f"image/{suffix}"

    with path.open("rb") as file:
        encoded = base64.b64encode(file.read()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def call_model(
    prompt: str,
    image_paths: list[str] | None = None,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.2,
) -> str:
    """Call a multimodal OpenAI model and return text output.

    Args:
        prompt: Fully composed prompt text.
        image_paths: Optional list of image paths for multimodal input.
        model: Model name, defaults to config.DEFAULT_MODEL.
        temperature: Sampling temperature.

    Returns:
        Model output as plain text.
    """

    client = _get_client()

    content: list[dict[str, Any]] = [{"type": "input_text", "text": prompt}]
    for image_path in image_paths or []:
        image_url = _encode_image_to_data_url(image_path)
        content.append({"type": "input_image", "image_url": image_url})

    response = client.responses.create(
        model=model,
        temperature=temperature,
        input=[
            {
                "type": "message",
                "role": "user",
                "content": content,
            }
        ],
    )

    output_text = response.output_text
    if output_text:
        return output_text.strip()

    # Fallback for unexpected response formatting.
    try:
        return json.dumps(response.model_dump(), indent=2)
    except Exception as exc:  # pragma: no cover
        raise OpenAIClientError("Model returned no text output.") from exc
