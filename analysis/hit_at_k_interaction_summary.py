"""Interaction-level Hit@k summary statistics.

Analysis 2: pool binary per-interaction Hit@k values across participants, then
compute the mean and sample standard deviation for each metric and k.
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
OUTPUT_PATH = ROOT / "analysis" / "hit_at_k_interaction_summary.csv"


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

    per_interaction = payload["per_interaction"]
    if not isinstance(per_interaction, list):
        raise ValueError(f"{path.name} per_interaction must be a list")

    blocks_by_k = {int(block.get("k")): block for block in per_interaction if isinstance(block, dict)}
    if sorted(blocks_by_k) != K_VALUES:
        raise ValueError(f"{path.name} has per_interaction k blocks={sorted(blocks_by_k)}, expected {K_VALUES}")

    total_interactions = int(payload.get("total_interactions", 0))
    for k, block in blocks_by_k.items():
        interactions = block.get("interactions")
        if not isinstance(interactions, list):
            raise ValueError(f"{path.name} per_interaction k={k} interactions must be a list")
        if len(interactions) != total_interactions:
            raise ValueError(
                f"{path.name} k={k} has {len(interactions)} interaction rows, "
                f"expected total_interactions={total_interactions}"
            )
        hit_key = f"hit_at_{k}"
        for item in interactions:
            hit_block = item.get(hit_key) if isinstance(item, dict) else None
            if not isinstance(hit_block, dict):
                raise ValueError(f"{path.name} interaction row is missing {hit_key}")
            for metric in METRICS:
                if metric not in hit_block:
                    raise ValueError(f"{path.name} interaction row is missing {hit_key}.{metric}")


def main() -> None:
    interaction_rows: list[dict[str, Any]] = []

    for path in load_hit_at_k_files():
        payload = json.loads(path.read_text(encoding="utf-8"))
        validate_payload(payload, path)
        pid = participant_id(path)

        for block in payload["per_interaction"]:
            k = int(block["k"])
            hit_key = f"hit_at_{k}"
            for item in block["interactions"]:
                for metric, metric_label in METRICS.items():
                    interaction_rows.append(
                        {
                            "participant": pid,
                            "interaction": item.get("interaction"),
                            "k": k,
                            "metric": metric,
                            "metric_label": metric_label,
                            "hit": item[hit_key][metric],
                        }
                    )

    interaction_df = pd.DataFrame(interaction_rows)
    summary_df = (
        interaction_df.groupby(["k", "metric", "metric_label"], sort=False)["hit"]
        .agg(n_interactions="count", mean_hit_at_k="mean", std_hit_at_k=lambda x: x.std(ddof=1))
        .reset_index()
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(OUTPUT_PATH, index=False)

    participants = sorted(interaction_df["participant"].unique(), key=lambda pid: int(pid[1:]))
    print(
        f"Saved interaction summary for {len(participants)} participants "
        f"({', '.join(participants)}) to {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
