"""Participant-level Hit@k boxplots.

Analysis 3: plot participant average Hit@k distributions. Each subplot is one
metric, and each box is the distribution of participant averages for k=1,3,10.
"""

from __future__ import annotations

import json
import os
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
OUTPUT_CSV_PATH = ROOT / "analysis" / "hit_at_k_participant_long.csv"
OUTPUT_PNG_PATH = ROOT / "analysis" / "hit_at_k_participant_boxplots.png"
MPLCONFIGDIR = Path("/private/tmp") / "inst878-vqasim-matplotlib-cache"


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


def build_participant_dataframe() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for path in load_hit_at_k_files():
        payload = json.loads(path.read_text(encoding="utf-8"))
        validate_payload(payload, path)
        pid = participant_id(path)

        for k in K_VALUES:
            for metric, metric_label in METRICS.items():
                rows.append(
                    {
                        "participant": pid,
                        "k": k,
                        "metric": metric,
                        "metric_label": metric_label,
                        "hit_at_k": payload["averages"][str(k)][metric]["hit_at_k"],
                    }
                )

    return pd.DataFrame(rows)


def main() -> None:
    participant_df = build_participant_dataframe()

    OUTPUT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    participant_df.to_csv(OUTPUT_CSV_PATH, index=False)

    MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(15, 4.5), sharey=True)
    colors = ["#5B8DEF", "#39A96B", "#E07A5F"]

    for ax, (metric, metric_label) in zip(axes, METRICS.items()):
        data = [
            participant_df[
                (participant_df["metric"] == metric) & (participant_df["k"] == k)
            ]["hit_at_k"].tolist()
            for k in K_VALUES
        ]
        box = ax.boxplot(
            data,
            tick_labels=[f"Hit@{k}" for k in K_VALUES],
            patch_artist=True,
            widths=0.55,
            showmeans=True,
            meanprops={
                "marker": "o",
                "markerfacecolor": "#222222",
                "markeredgecolor": "#222222",
                "markersize": 4,
            },
            medianprops={"color": "#222222", "linewidth": 1.5},
            whiskerprops={"color": "#555555"},
            capprops={"color": "#555555"},
        )
        for patch, color in zip(box["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.72)
            patch.set_edgecolor("#444444")

        ax.set_title(metric_label)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.25)
        ax.set_xlabel("k")

    axes[0].set_ylabel("Participant average Hit@k")
    fig.suptitle("Participant Average Hit@k by Metric", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUTPUT_PNG_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)

    participants = sorted(participant_df["participant"].unique(), key=lambda pid: int(pid[1:]))
    print(
        f"Saved participant boxplots for {len(participants)} participants "
        f"({', '.join(participants)}) to {OUTPUT_PNG_PATH}; data to {OUTPUT_CSV_PATH}"
    )


if __name__ == "__main__":
    main()
