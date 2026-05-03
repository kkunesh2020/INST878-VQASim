"""Distribution of matched ground-truth turns for any-turn target matches.

Analysis 6: pool successful interactions across participants and count the
ground-truth turn indices matched by generated responses.
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


THRESHOLD = 0.5

ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "aggregated-outputs"
OUTPUT_CSV_PATH = ROOT / "analysis" / "hit_at_k_turn_distribution_counts.csv"
OUTPUT_PNG_PATH = ROOT / "analysis" / "hit_at_k_turn_distribution.png"
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


def parse_turn(value: Any, *, path: Path, interaction: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{path.name} interaction {interaction} has invalid matched_user_turn: {value}"
        ) from exc


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


def count_matched_turns() -> tuple[pd.DataFrame, list[str], int, int, bool]:
    last_turn_counts: Counter[int] = Counter()
    not_last_turn_counts: Counter[int] = Counter()
    participants: list[str] = []
    successful_interactions = 0
    score_matched_rows = 0
    non_null_turn_rows = 0

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
            turn_targets = item.get("ground_truth", {}).get("turn_targets")
            if not isinstance(turn_targets, list) or not turn_targets:
                raise ValueError(
                    f"{path.name} interaction {item.get('interaction')} "
                    "is missing non-empty ground_truth.turn_targets"
                )
            last_turn = len(turn_targets)

            successful_interactions += 1
            for response in generated:
                if not isinstance(response, dict):
                    continue

                turn = parse_turn(
                    response.get("matched_user_turn"),
                    path=path,
                    interaction=item.get("interaction"),
                )
                score_matches = safe_float(response.get("matched_user_turn_score")) >= THRESHOLD

                if turn is not None:
                    non_null_turn_rows += 1
                if score_matches:
                    score_matched_rows += 1

                if score_matches and turn is not None:
                    if turn < 1 or turn > last_turn:
                        raise ValueError(
                            f"{path.name} interaction {item.get('interaction')} "
                            f"has matched_user_turn={turn}, but last turn is {last_turn}"
                        )
                    if turn == last_turn:
                        last_turn_counts[turn] += 1
                    else:
                        not_last_turn_counts[turn] += 1

    turns = sorted(set(last_turn_counts) | set(not_last_turn_counts))
    rows = [
        {
            "matched_user_turn": turn,
            "last_turn_count": last_turn_counts[turn],
            "not_last_turn_count": not_last_turn_counts[turn],
            "match_count": last_turn_counts[turn] + not_last_turn_counts[turn],
        }
        for turn in turns
    ]
    total_matches = sum(last_turn_counts.values()) + sum(not_last_turn_counts.values())
    filters_equivalent = score_matched_rows == non_null_turn_rows == total_matches
    return pd.DataFrame(rows), participants, successful_interactions, total_matches, filters_equivalent


def plot_turn_distribution(counts_df: pd.DataFrame) -> None:
    MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    ax.bar(
        counts_df["matched_user_turn"],
        counts_df["last_turn_count"],
        width=0.75,
        color="#80cdc1",
        edgecolor="#444444",
        linewidth=0.6,
        label="Last turn",
    )
    ax.bar(
        counts_df["matched_user_turn"],
        counts_df["not_last_turn_count"],
        width=0.75,
        bottom=counts_df["last_turn_count"],
        color="#dfc27d",
        edgecolor="#444444",
        linewidth=0.6,
        label="Not last turn",
    )
    ax.set_xlabel("Matched ground-truth turn")
    ax.set_ylabel("Number of matches")
    ax.set_xticks(counts_df["matched_user_turn"].tolist())
    ax.legend(frameon=True)

    max_count = int(counts_df["match_count"].max()) if not counts_df.empty else 0
    ax.set_ylim(0, max_count + max(1, int(max_count * 0.1)))
    fig.tight_layout()
    fig.savefig(OUTPUT_PNG_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    counts_df, participants, successful_interactions, total_matches, filters_equivalent = count_matched_turns()

    OUTPUT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    counts_df.to_csv(OUTPUT_CSV_PATH, index=False)
    plot_turn_distribution(counts_df)

    equivalence_note = "yes" if filters_equivalent else "no"
    print(
        f"Saved turn distribution for {len(participants)} participants and "
        f"{successful_interactions} successful interactions. "
        f"Matched rows counted: {total_matches}. "
        f"Score and non-null filters equivalent: {equivalence_note}. "
        f"CSV: {OUTPUT_CSV_PATH}; plot: {OUTPUT_PNG_PATH}"
    )


if __name__ == "__main__":
    main()
