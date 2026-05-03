"""Rank distribution of category, target, and any-turn target matches.

Analysis 5: pool all successful interactions across participants and count
which ranks match the ground truth for each metric.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd


RANKS = list(range(1, 11))
THRESHOLD = 0.5
METRICS = {
    "category": {
        "label": "Category",
        "score_field": "task_type_match_score",
        "color": "#a6611a",
    },
    "target": {
        "label": "Target",
        "score_field": "target_match_score",
        "color": "#01665e",
    },
    "matched_user_turn": {
        "label": "Any-Turn Target",
        "score_field": "matched_user_turn_score",
        "color": "#5ab4ac",
    },
}

ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "aggregated-outputs"
OUTPUT_CSV_PATH = ROOT / "analysis" / "hit_at_k_rank_distribution_counts.csv"
OUTPUT_CATEGORY_PNG_PATH = ROOT / "analysis" / "hit_at_k_rank_distribution_category.png"
OUTPUT_TARGETS_PNG_PATH = ROOT / "analysis" / "hit_at_k_rank_distribution_targets.png"
OUTPUT_LINES_PNG_PATH = ROOT / "analysis" / "hit_at_k_rank_distribution_lines.png"
MPLCONFIGDIR = Path("/private/tmp") / "inst878-vqasim-matplotlib-cache"


def participant_id(path: Path) -> str:
    match = re.match(r"(P\d+)_diary_batch_comparison\.json$", path.name)
    if not match:
        raise ValueError(f"Unexpected input filename: {path.name}")
    return match.group(1)


def participant_sort_key(path: Path) -> int:
    return int(participant_id(path)[1:])


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def load_comparison_files() -> list[Path]:
    files = sorted(
        INPUT_DIR.glob("P*_diary_batch_comparison.json"),
        key=participant_sort_key,
    )
    if not files:
        raise FileNotFoundError(f"No comparison files found in {INPUT_DIR}")
    return files


def validate_payload(payload: dict[str, Any], path: Path) -> list[dict[str, Any]]:
    comparison_results = payload.get("comparison_results")
    if not isinstance(comparison_results, list):
        raise ValueError(f"{path.name} is missing list key: comparison_results")
    return comparison_results


def count_rank_matches() -> tuple[pd.DataFrame, list[str], int]:
    counts = {(metric, rank): 0 for metric in METRICS for rank in RANKS}
    participants: list[str] = []
    successful_interactions = 0

    for path in load_comparison_files():
        payload = json.loads(path.read_text(encoding="utf-8"))
        comparison_results = validate_payload(payload, path)
        participants.append(participant_id(path))

        for item in comparison_results:
            if not isinstance(item, dict):
                continue
            if str(item.get("status", "")).lower().strip() != "success":
                continue

            generated = item.get("generated_responses")
            if not isinstance(generated, list):
                raise ValueError(
                    f"{path.name} interaction {item.get('interaction')} "
                    "is missing list key: generated_responses"
                )

            successful_interactions += 1
            for response in generated:
                if not isinstance(response, dict):
                    continue
                rank = safe_int(response.get("rank"))
                if rank not in RANKS:
                    raise ValueError(
                        f"{path.name} interaction {item.get('interaction')} "
                        f"has invalid rank: {response.get('rank')}"
                    )

                for metric, spec in METRICS.items():
                    if safe_float(response.get(spec["score_field"])) >= THRESHOLD:
                        counts[(metric, rank)] += 1

    rows: list[dict[str, Any]] = []
    for metric, spec in METRICS.items():
        for rank in RANKS:
            rows.append(
                {
                    "metric": metric,
                    "metric_label": spec["label"],
                    "rank": rank,
                    "match_count": counts[(metric, rank)],
                }
            )

    return pd.DataFrame(rows), participants, successful_interactions


def draw_rank_bars(ax: Any, counts_df: pd.DataFrame, metric: str) -> None:
    spec = METRICS[metric]
    metric_df = counts_df[counts_df["metric"] == metric].sort_values("rank")
    ax.bar(
        metric_df["rank"],
        metric_df["match_count"],
        width=1.0,
        align="center",
        color=spec["color"],
        edgecolor="#444444",
        linewidth=0.6,
    )
    ax.set_title(spec["label"])
    ax.set_xlabel("Rank")
    ax.set_xticks(RANKS)
    ax.set_xlim(-1, 12)


def plot_category_distribution(counts_df: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    max_count = int(counts_df[counts_df["metric"] == "category"]["match_count"].max())
    draw_rank_bars(ax, counts_df, "category")
    ax.set_ylabel("Number of matches")
    ax.set_ylim(0, max_count + max(1, int(max_count * 0.1)))
    fig.tight_layout()
    fig.savefig(OUTPUT_CATEGORY_PNG_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_target_distributions(counts_df: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    metrics = ["target", "matched_user_turn"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), sharey=True)
    max_count = int(counts_df[counts_df["metric"].isin(metrics)]["match_count"].max())

    for ax, metric in zip(axes, metrics):
        draw_rank_bars(ax, counts_df, metric)

    axes[0].set_ylabel("Number of matches")
    axes[0].set_ylim(0, max_count + max(1, int(max_count * 0.1)))
    fig.tight_layout()
    fig.savefig(OUTPUT_TARGETS_PNG_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_line_distribution(counts_df: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    markers = {
        "category": "o",
        "target": "s",
        "matched_user_turn": "x",
    }
    fig, ax = plt.subplots(figsize=(7.2, 4.4))

    for metric, spec in METRICS.items():
        metric_df = counts_df[counts_df["metric"] == metric].sort_values("rank")
        ax.plot(
            metric_df["rank"],
            metric_df["match_count"],
            color=spec["color"],
            marker=markers[metric],
            markersize=5,
            linewidth=1.0,
            label=spec["label"],
        )

    max_count = int(counts_df["match_count"].max())
    ax.set_xlabel("Rank")
    ax.set_ylabel("Number of matches")
    ax.set_xticks(RANKS)
    ax.set_xlim(0.75, 10.25)
    ax.set_ylim(0, max_count + max(1, int(max_count * 0.1)))
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(OUTPUT_LINES_PNG_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_rank_distribution(counts_df: pd.DataFrame) -> None:
    MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

    plot_category_distribution(counts_df)
    plot_target_distributions(counts_df)
    plot_line_distribution(counts_df)


def main() -> None:
    counts_df, participants, successful_interactions = count_rank_matches()

    OUTPUT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    counts_df.to_csv(OUTPUT_CSV_PATH, index=False)
    plot_rank_distribution(counts_df)

    print(
        f"Saved rank distribution for {len(participants)} participants and "
        f"{successful_interactions} successful interactions to {OUTPUT_CSV_PATH}; "
        f"plots to {OUTPUT_CATEGORY_PNG_PATH}, {OUTPUT_TARGETS_PNG_PATH}, "
        f"and {OUTPUT_LINES_PNG_PATH}"
    )


if __name__ == "__main__":
    main()
