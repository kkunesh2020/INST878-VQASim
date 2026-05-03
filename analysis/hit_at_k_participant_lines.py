"""Participant-level Hit@k line distributions.

Analysis 7: for each participant, plot four metric-specific Hit@k trajectories
connecting Hit@1, Hit@3, and Hit@10.
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
METRIC_COLORS = {
    "category": "#dfc27d",
    "target": "#80cdc1",
    "combined": "#a6611a",
    "matched_user_turn": "#018571",
}
K_MARKERS = {
    1: "o",
    3: "s",
    10: "^",
}

ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "aggregated-outputs"
OUTPUT_CSV_PATH = ROOT / "analysis" / "hit_at_k_participant_lines_data.csv"
OUTPUT_PNG_PATH = ROOT / "analysis" / "hit_at_k_participant_lines.png"
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


def plot_participant_lines(participant_df: pd.DataFrame) -> None:
    MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    participants = sorted(participant_df["participant"].unique(), key=lambda pid: int(pid[1:]))
    metric_offsets = {
        "category": 0.24,
        "target": 0.08,
        "combined": -0.08,
        "matched_user_turn": -0.24,
    }
    participant_y = {participant: index for index, participant in enumerate(reversed(participants))}

    fig, ax = plt.subplots(figsize=(10, 7))

    for participant in participants:
        base_y = participant_y[participant]
        for metric in METRICS:
            metric_df = participant_df[
                (participant_df["participant"] == participant) & (participant_df["metric"] == metric)
            ].sort_values("k")
            y = base_y + metric_offsets[metric]
            ax.plot(
                metric_df["hit_at_k"],
                [y] * len(metric_df),
                color=METRIC_COLORS[metric],
                linewidth=1.4,
                zorder=2,
            )
            for row in metric_df.itertuples(index=False):
                ax.scatter(
                    row.hit_at_k,
                    y,
                    marker=K_MARKERS[int(row.k)],
                    s=34,
                    color=METRIC_COLORS[metric],
                    edgecolor="#333333",
                    linewidth=0.5,
                    zorder=3,
                )

    metric_handles = [
        Line2D([0], [0], color=METRIC_COLORS[metric], linewidth=2, label=label)
        for metric, label in METRICS.items()
    ]
    k_handles = [
        Line2D(
            [0],
            [0],
            marker=K_MARKERS[k],
            color="#333333",
            linestyle="None",
            markerfacecolor="#f5f5f5",
            markeredgecolor="#333333",
            markersize=6,
            label=f"Hit@{k}",
        )
        for k in K_VALUES
    ]
    fig.legend(
        handles=metric_handles,
        loc="lower center",
        bbox_to_anchor=(0.673, 0.14),
        ncol=4,
        # title="Metric",
        frameon=True,
    )
    ax.legend(handles=k_handles, loc="upper right", frameon=True)

    ax.set_yticks([participant_y[participant] for participant in participants])
    ax.set_yticklabels(participants)
    ax.set_ylim(-0.7, len(participants) - 0.3)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Participant average Hit@k")
    ax.set_ylabel("Participant")
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(OUTPUT_PNG_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    participant_df = build_participant_dataframe()

    OUTPUT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    participant_df.to_csv(OUTPUT_CSV_PATH, index=False)
    plot_participant_lines(participant_df)

    participants = sorted(participant_df["participant"].unique(), key=lambda pid: int(pid[1:]))
    print(
        f"Saved participant Hit@k lines for {len(participants)} participants "
        f"({', '.join(participants)}) to {OUTPUT_PNG_PATH}; data to {OUTPUT_CSV_PATH}"
    )


if __name__ == "__main__":
    main()
