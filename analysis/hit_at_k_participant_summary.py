"""Participant-level Hit@k summary statistics.

Analysis 1: compute the mean and sample standard deviation of participant
average Hit@k values for category, target, combined, and any-turn target.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


K_VALUES = [1, 3, 10]
METRICS = {
    "category": "Category",
    "target": "Target",
    "combined": "Category + Target",
    "matched_user_turn": "Any-Turn Target",
}

ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "aggregated-outputs"
OUTPUT_PATH = ROOT / "analysis" / "hit_at_k_participant_summary.csv"


def participant_id(path: Path) -> str:
    match = re.match(r"(P\d+)_diary_batch_comparison_hit_at_k\.json$", path.name)
    if not match:
        raise ValueError(f"Unexpected input filename: {path.name}")
    return match.group(1)


def participant_sort_key(path: Path) -> int:
    return int(participant_id(path)[1:])


def load_hit_at_k_files() -> list[Path]:
    files = sorted(
        INPUT_DIR.glob("P*_diary_batch_comparison_hit_at_k.json"),
        key=participant_sort_key,
    )
    if not files:
        raise FileNotFoundError(f"No hit@k files found in {INPUT_DIR}")
    return files


def validate_payload(payload: dict[str, Any], path: Path) -> None:
    for key in ("averages", "per_interaction", "k_values"):
        if key not in payload:
            raise ValueError(f"{path.name} is missing required key: {key}")

    found_k_values = sorted(int(k) for k in payload["k_values"])
    if found_k_values != K_VALUES:
        raise ValueError(f"{path.name} has k_values={found_k_values}, expected {K_VALUES}")

    averages = payload["averages"]
    for k in K_VALUES:
        k_block = averages.get(str(k))
        if not isinstance(k_block, dict):
            raise ValueError(f"{path.name} is missing averages for k={k}")
        for metric in METRICS:
            metric_block = k_block.get(metric)
            if not isinstance(metric_block, dict) or "hit_at_k" not in metric_block:
                raise ValueError(f"{path.name} is missing averages[{k}][{metric}].hit_at_k")


def main() -> None:
    participant_rows: list[dict[str, Any]] = []

    for path in load_hit_at_k_files():
        payload = json.loads(path.read_text(encoding="utf-8"))
        validate_payload(payload, path)
        pid = participant_id(path)

        for k in K_VALUES:
            for metric, metric_label in METRICS.items():
                participant_rows.append(
                    {
                        "participant": pid,
                        "k": k,
                        "metric": metric,
                        "metric_label": metric_label,
                        "hit_at_k": payload["averages"][str(k)][metric]["hit_at_k"],
                    }
                )

    participant_df = pd.DataFrame(participant_rows)
    summary_df = (
        participant_df.groupby(["k", "metric", "metric_label"], sort=False)["hit_at_k"]
        .agg(n_participants="count", mean_hit_at_k="mean", std_hit_at_k=lambda x: x.std(ddof=1))
        .reset_index()
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(OUTPUT_PATH, index=False)

    participants = sorted(participant_df["participant"].unique(), key=lambda pid: int(pid[1:]))
    print(
        f"Saved participant summary for {len(participants)} participants "
        f"({', '.join(participants)}) to {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
