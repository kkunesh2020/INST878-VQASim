"""Configuration and path management for the BLV VQA multi-agent pipeline."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent
PROMPTS_DIR = PROJECT_ROOT / "prompts"
OUTPUT_JSON_DIR = PROJECT_ROOT / "outputs" / "json"
OUTPUT_READABLE_DIR = PROJECT_ROOT / "outputs" / "readable"
PARTICIPANT_DATA_DIR = PROJECT_ROOT / "participant_data"

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Ensure output directories exist at runtime.
OUTPUT_JSON_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_READABLE_DIR.mkdir(parents=True, exist_ok=True)
