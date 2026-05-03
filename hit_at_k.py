"""Compute Hit@k metrics from a batch comparison JSON file.

Metrics:
- Category Hit@k: any top-k item with task_type_match_score >= threshold
- Target Hit@k: any top-k item with target_match_score >= threshold
- Combined Hit@k: any top-k item where both scores >= threshold
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _top_k_rows(generated_responses: list[dict[str, Any]], k: int) -> list[dict[str, Any]]:
    sortable: list[tuple[int, dict[str, Any]]] = []
    for idx, row in enumerate(generated_responses, start=1):
        if not isinstance(row, dict):
            continue
        rank = _safe_int(row.get("rank"), idx)
        sortable.append((rank, row))

    sortable.sort(key=lambda item: item[0])
    return [row for _, row in sortable[:k]]


def _compute_hit_flags(rows: list[dict[str, Any]], threshold: float) -> tuple[bool, bool, bool, bool]:
    category_hit = False
    target_hit = False
    combined_hit = False
    matched_turn_hit = False

    for row in rows:
        category_ok = _safe_float(row.get("task_type_match_score", 0)) >= threshold
        target_ok = _safe_float(row.get("target_match_score", 0)) >= threshold
        matched_turn_ok = _safe_float(row.get("matched_user_turn_score", 0)) >= threshold

        category_hit = category_hit or category_ok
        target_hit = target_hit or target_ok
        combined_hit = combined_hit or (category_ok and target_ok)
        matched_turn_hit = matched_turn_hit or matched_turn_ok

    return category_hit, target_hit, combined_hit, matched_turn_hit


def compute_hit_at_k(
    batch_payload: dict[str, Any],
    ks: list[int],
    threshold: float,
    success_only: bool = True,
) -> dict[str, Any]:
    comparison_results = batch_payload.get("comparison_results", [])
    if not isinstance(comparison_results, list):
        comparison_results = []

    interactions: list[dict[str, Any]] = []
    for item in comparison_results:
        if not isinstance(item, dict):
            continue
        if success_only and str(item.get("status", "")).lower().strip() != "success":
            continue
        generated = item.get("generated_responses", [])
        if not isinstance(generated, list):
            generated = []
        interactions.append(
            {
                "interaction": item.get("interaction"),
                "generated_responses": generated,
            }
        )

    metrics: dict[str, Any] = {
        "total_interactions": len(interactions),
        "threshold": threshold,
        "k_values": ks,
        "per_interaction": [],
        "averages": {},
    }

    for k in ks:
        interaction_scores: list[dict[str, Any]] = []
        for interaction in interactions:
            rows = _top_k_rows(interaction["generated_responses"], k)
            category_hit, target_hit, combined_hit, matched_turn_hit = _compute_hit_flags(rows, threshold)

            interaction_scores.append(
                {
                    "interaction": interaction["interaction"],
                    "category_hit": 1 if category_hit else 0,
                    "target_hit": 1 if target_hit else 0,
                    "combined_hit": 1 if combined_hit else 0,
                    "matched_user_turn_hit": 1 if matched_turn_hit else 0,
                }
            )

        total = len(interaction_scores)
        denom = total if total > 0 else 1
        category_hits = sum(item["category_hit"] for item in interaction_scores)
        target_hits = sum(item["target_hit"] for item in interaction_scores)
        combined_hits = sum(item["combined_hit"] for item in interaction_scores)
        matched_turn_hits = sum(item["matched_user_turn_hit"] for item in interaction_scores)

        for item in interaction_scores:
            item[f"hit_at_{k}"] = {
                "category": item.pop("category_hit"),
                "target": item.pop("target_hit"),
                "combined": item.pop("combined_hit"),
                "matched_user_turn": item.pop("matched_user_turn_hit"),
            }

        metrics["per_interaction"].append(
            {
                "k": k,
                "interactions": interaction_scores,
            }
        )

        metrics["averages"][str(k)] = {
            "category": {
                "hits": category_hits,
                "total": total,
                "hit_at_k": category_hits / denom,
            },
            "target": {
                "hits": target_hits,
                "total": total,
                "hit_at_k": target_hits / denom,
            },
            "combined": {
                "hits": combined_hits,
                "total": total,
                "hit_at_k": combined_hits / denom,
            },
            "matched_user_turn": {
                "hits": matched_turn_hits,
                "total": total,
                "hit_at_k": matched_turn_hits / denom,
            },
        }

    return metrics


def _parse_k_values(raw: str) -> list[int]:
    values: list[int] = []
    for part in raw.split(","):
        token = part.strip()
        if not token:
            continue
        value = int(token)
        if value < 1:
            raise ValueError("All k values must be >= 1.")
        values.append(value)

    if not values:
        raise ValueError("At least one k value is required.")
    return values


def _print_summary(metrics: dict[str, Any]) -> None:
    total = metrics.get("total_interactions", 0)
    threshold = metrics.get("threshold", 0.5)
    print(f"Interactions evaluated: {total}")
    print(f"Partial-match threshold: {threshold}")
    print()

    per_interaction = metrics.get("per_interaction", [])
    if isinstance(per_interaction, list):
        for block in per_interaction:
            if not isinstance(block, dict):
                continue
            k = block.get("k")
            interactions = block.get("interactions", [])
            print(f"Per-interaction Hit@{k}:")
            for item in interactions:
                if not isinstance(item, dict):
                    continue
                hit_vals = item.get(f"hit_at_{k}", {})
                print(
                    f"  interaction {item.get('interaction')}: "
                    f"category={hit_vals.get('category', 0)} "
                    f"target={hit_vals.get('target', 0)} "
                    f"combined={hit_vals.get('combined', 0)} "
                    f"matched_user_turn={hit_vals.get('matched_user_turn', 0)}"
                )
            print()

    averages = metrics.get("averages", {})
    for k in metrics.get("k_values", []):
        block = averages.get(str(k), {})
        category = block.get("category", {})
        target = block.get("target", {})
        combined = block.get("combined", {})
        matched_turn = block.get("matched_user_turn", {})

        print(f"Averaged Hit@{k} across interactions")
        print(
            f"  Category Hit@{k}: {category.get('hit_at_k', 0):.4f} "
            f"({category.get('hits', 0)}/{category.get('total', 0)})"
        )
        print(
            f"  Target Hit@{k}:   {target.get('hit_at_k', 0):.4f} "
            f"({target.get('hits', 0)}/{target.get('total', 0)})"
        )
        print(
            f"  Combined Hit@{k}: {combined.get('hit_at_k', 0):.4f} "
            f"({combined.get('hits', 0)}/{combined.get('total', 0)})"
        )
        print(
            f"  Matched User Turn Hit@{k}: {matched_turn.get('hit_at_k', 0):.4f} "
            f"({matched_turn.get('hits', 0)}/{matched_turn.get('total', 0)})"
        )
        print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute Category/Target/Combined Hit@k from a batch comparison JSON file."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to batch comparison JSON (e.g., outputs2/json/P2_diary_batch_comparison.json).",
    )
    parser.add_argument(
        "--k-values",
        default="1,3,10",
        help="Comma-separated k values (default: 1,3,10).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Match threshold for successful retrieval (default: 0.5).",
    )
    parser.add_argument(
        "--include-failed",
        action="store_true",
        help="Include failed interactions in denominator.",
    )
    parser.add_argument(
        "--output",
        default="",
        help=(
            "Path to save metrics JSON. If omitted, saves next to input as "
            "<input_stem>_hit_at_k.json."
        ),
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    k_values = _parse_k_values(args.k_values)
    if args.threshold < 0:
        raise SystemExit("--threshold must be >= 0")

    payload = json.loads(input_path.read_text(encoding="utf-8"))
    metrics = compute_hit_at_k(
        batch_payload=payload,
        ks=k_values,
        threshold=args.threshold,
        success_only=not args.include_failed,
    )

    _print_summary(metrics)

    output_path = Path(args.output) if args.output else input_path.with_name(f"{input_path.stem}_hit_at_k.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"Saved metrics JSON: {output_path}")


if __name__ == "__main__":
    main()
