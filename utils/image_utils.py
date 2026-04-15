"""Image utilities for path validation and lightweight metadata extraction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image


def validate_image_paths(image_paths: list[str]) -> tuple[list[str], list[str]]:
    """Validate a list of image paths.

    Returns:
        A tuple of (existing_paths, missing_paths).
    """

    existing: list[str] = []
    missing: list[str] = []

    for image_path in image_paths:
        if Path(image_path).exists():
            existing.append(image_path)
        else:
            missing.append(image_path)

    return existing, missing


def image_metadata(image_path: str) -> dict[str, Any]:
    """Return basic metadata for an image path."""

    path = Path(image_path)
    with Image.open(path) as image:
        return {
            "path": str(path),
            "format": image.format,
            "mode": image.mode,
            "width": image.width,
            "height": image.height,
        }
