"""Participant-level Hit@k gap analysis with Wilcoxon signed-rank tests.

Analysis 4: compare participant average Hit@k values across k values.
Gaps are computed per participant and metric.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd
from scipy.stats import wilcoxon


K_VALUES = [1, 3, 10]
METRICS = {
    "category": "Category",
    "target": "Target",
    "combined": "Category + Target",
    "matched_user_turn": "Any-Turn Target",
}
COMPARISONS = {
    "1_to_3": (1, 3),
    "1_to_10": (1, 10),
    "3_to_10": (3, 10),
}
ALTERNATIVE = "greater"

ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "aggregated-outputs"
OUTPUT_GAPS_PATH = ROOT / "analysis" / "hit_at_k_participant_gaps.csv"
OUTPUT_SUMMARY_PATH = ROOT / "analysis" / "hit_at_k_gap_wilcoxon.csv"


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


def build_participant_hit_dataframe() -> pd.DataFrame:
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


def build_gap_dataframe(hit_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for (participant, metric, metric_label), group in hit_df.groupby(
        ["participant", "metric", "metric_label"], sort=False
    ):
        values = {int(row.k): float(row.hit_at_k) for row in group.itertuples(index=False)}
        if sorted(values) != K_VALUES:
            raise ValueError(f"{participant} {metric} is missing one or more required k values")

        for comparison, (baseline_k, higher_k) in COMPARISONS.items():
            baseline = values[baseline_k]
            higher = values[higher_k]
            rows.append(
                {
                    "participant": participant,
                    "metric": metric,
                    "metric_label": metric_label,
                    "comparison": comparison,
                    "baseline_k": baseline_k,
                    "higher_k": higher_k,
                    "baseline_hit_at_k": baseline,
                    "higher_hit_at_k": higher,
                    "gap": higher - baseline,
                }
            )

    return pd.DataFrame(rows)


def run_wilcoxon(gaps: pd.Series) -> tuple[float, float, str]:
    if (gaps == 0).all():
        return math.nan, math.nan, "all paired differences are zero; Wilcoxon test not run"

    result = wilcoxon(gaps, alternative=ALTERNATIVE, zero_method="wilcox")
    return float(result.statistic), float(result.pvalue), ""


def build_summary_dataframe(gap_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for (metric, metric_label, comparison), group in gap_df.groupby(
        ["metric", "metric_label", "comparison"], sort=False
    ):
        gaps = group["gap"]
        statistic, p_value, note = run_wilcoxon(gaps)
        rows.append(
            {
                "metric": metric,
                "metric_label": metric_label,
                "comparison": comparison,
                "n_participants": int(gaps.count()),
                "mean_gap": float(gaps.mean()),
                "std_gap": float(gaps.std(ddof=1)),
                "median_gap": float(gaps.median()),
                "wilcoxon_statistic": statistic,
                "p_value": p_value,
                "alternative": ALTERNATIVE,
                "note": note,
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    hit_df = build_participant_hit_dataframe()
    gap_df = build_gap_dataframe(hit_df)
    summary_df = build_summary_dataframe(gap_df)

    OUTPUT_GAPS_PATH.parent.mkdir(parents=True, exist_ok=True)
    gap_df.to_csv(OUTPUT_GAPS_PATH, index=False)
    summary_df.to_csv(OUTPUT_SUMMARY_PATH, index=False)

    participants = sorted(hit_df["participant"].unique(), key=lambda pid: int(pid[1:]))
    print(
        f"Saved gap analysis for {len(participants)} participants "
        f"({', '.join(participants)}) to {OUTPUT_SUMMARY_PATH}; "
        f"participant gaps to {OUTPUT_GAPS_PATH}"
    )


if __name__ == "__main__":
    main()
