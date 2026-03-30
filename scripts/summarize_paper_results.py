from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


FAMILY_ORDER = [
    "Physical Maintenance",
    "Environmental Amenity",
    "Visual Legibility",
    "Mobility Infrastructure",
]

CITY_ORDER = [
    "Amsterdam",
    "Abuja",
    "San Francisco",
    "Santiago",
    "Singapore",
]

CONCEPT_ORDER = [
    ("Physical Maintenance", "graffiti removal"),
    ("Physical Maintenance", "litter removal"),
    ("Physical Maintenance", "facade repair"),
    ("Physical Maintenance", "surface cleaning"),
    ("Physical Maintenance", "shutter repair"),
    ("Environmental Amenity", "localized greenery addition"),
    ("Environmental Amenity", "lighting repair"),
    ("Environmental Amenity", "tree canopy management"),
    ("Visual Legibility", "signage decluttering"),
    ("Visual Legibility", "storefront transparency increase"),
    ("Mobility Infrastructure", "crosswalk repainting"),
    ("Mobility Infrastructure", "lane marking repainting"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize candidate/scored paper outputs into table-ready JSON.")
    parser.add_argument("--manifest", required=True, help="Manifest CSV with image_id -> city.")
    parser.add_argument("--candidate-csv", required=True, help="Candidate CSV from generate_counterfactual.")
    parser.add_argument("--scored-csv", required=True, help="Scored CSV from run_analysis.")
    parser.add_argument("--summary-csv", required=True, help="Per-image summary CSV from run_analysis.")
    parser.add_argument("--scores-json", required=True, help="Baseline score cache JSON from prepare_specs.")
    parser.add_argument("--output", default=None, help="Optional output JSON path.")
    return parser.parse_args()


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


def coerce_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def maybe_float(value: Any) -> float | None:
    if value in {"", None}:
        return None
    return float(value)


def maybe_int(value: Any) -> int | None:
    if value in {"", None}:
        return None
    return int(float(value))


def safe_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(statistics.mean(values))


def safe_median(values: list[float]) -> float | None:
    if not values:
        return None
    return float(statistics.median(values))


def round_or_none(value: float | None, digits: int = 3) -> float | None:
    if value is None:
        return None
    return round(value, digits)


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float | None, float | None]:
    if total <= 0:
        return None, None
    p_hat = successes / total
    denom = 1 + (z**2 / total)
    center = (p_hat + (z**2 / (2 * total))) / denom
    margin = (z / denom) * math.sqrt((p_hat * (1 - p_hat) / total) + (z**2 / (4 * total**2)))
    return center - margin, center + margin


def rankdata(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg_rank
        i = j + 1
    return ranks


def pearsonr(xs: list[float], ys: list[float]) -> float:
    x_mean = statistics.mean(xs)
    y_mean = statistics.mean(ys)
    x_var = sum((x - x_mean) ** 2 for x in xs)
    y_var = sum((y - y_mean) ** 2 for y in ys)
    if x_var == 0 or y_var == 0:
        return 0.0
    cov = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    return cov / math.sqrt(x_var * y_var)


def compute_spearman(baselines: list[float], counts: list[int]) -> dict[str, float | None]:
    if len(baselines) < 2:
        return {"rho": None, "p_value": None}

    try:
        from scipy.stats import spearmanr  # type: ignore

        rho, pval = spearmanr(baselines, counts)
        return {
            "rho": float(rho) if rho == rho else None,
            "p_value": float(pval) if pval == pval else None,
        }
    except Exception:
        rho = pearsonr(rankdata(baselines), rankdata([float(value) for value in counts]))
        return {"rho": float(rho), "p_value": None}


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    manifest_rows = read_csv(args.manifest)
    all_candidate_rows = read_csv(args.candidate_csv)
    scored_rows = read_csv(args.scored_csv)
    per_image_rows = read_csv(args.summary_csv)
    baseline_scores = {str(k): float(v) for k, v in read_json(args.scores_json).items()}

    manifest_by_id = {row["image_id"]: row for row in manifest_rows}
    city_order = [city for city in CITY_ORDER if any(row["city"] == city for row in manifest_rows)]

    candidate_rows = [row for row in all_candidate_rows if row.get("candidate_id")]
    error_rows = [
        row for row in all_candidate_rows
        if not row.get("candidate_id") and str(row.get("critic_notes", "")).startswith("ERROR:")
    ]
    scored_candidate_rows = [row for row in scored_rows if row.get("candidate_id")]
    valid_rows = [row for row in scored_candidate_rows if coerce_bool(row.get("critic_is_valid"))]
    valid_deltas = [float(row["delta_classifier"]) for row in valid_rows if row.get("delta_classifier") not in {"", None}]
    overall_ci_low, overall_ci_high = wilson_interval(len(valid_rows), len(candidate_rows))

    overall = {
        "n_images": len(per_image_rows),
        "candidate_rows": len(candidate_rows),
        "valid_rows": len(valid_rows),
        "error_rows": len(error_rows),
        "images_with_candidate_generation_failure": len({
            Path(row["input_image_path"]).stem for row in error_rows if row.get("input_image_path")
        }),
        "images_with_any_valid": sum((maybe_int(row.get("n_valid")) or 0) > 0 for row in per_image_rows),
        "images_with_multiple_valid": sum((maybe_int(row.get("n_valid")) or 0) > 1 for row in per_image_rows),
        "mean_coverage": round_or_none(safe_mean([float(row["coverage"]) for row in per_image_rows])),
        "valid_rate": round_or_none(len(valid_rows) / len(candidate_rows) if candidate_rows else None),
        "valid_rate_ci_low": round_or_none(overall_ci_low),
        "valid_rate_ci_high": round_or_none(overall_ci_high),
        "mean_delta_aux": round_or_none(safe_mean(valid_deltas)),
        "median_delta_aux": round_or_none(safe_median(valid_deltas)),
        "min_delta_aux": round_or_none(min(valid_deltas) if valid_deltas else None),
        "max_delta_aux": round_or_none(max(valid_deltas) if valid_deltas else None),
        "n_aux_effective": sum(coerce_bool(row.get("exceeds_auxiliary_threshold")) for row in valid_rows),
    }

    family_table: list[dict[str, Any]] = []
    for family in FAMILY_ORDER:
        family_valid_deltas: list[float] = []
        family_prop = 0
        family_valid = 0
        for concept_family, concept in CONCEPT_ORDER:
            if concept_family != family:
                continue
            concept_rows = [row for row in scored_candidate_rows if row.get("lever_concept") == concept]
            concept_valid = [row for row in concept_rows if coerce_bool(row.get("critic_is_valid"))]
            concept_deltas = [
                float(row["delta_classifier"])
                for row in concept_valid
                if row.get("delta_classifier") not in {"", None}
            ]
            family_prop += len(concept_rows)
            family_valid += len(concept_valid)
            family_valid_deltas.extend(concept_deltas)
            concept_ci_low, concept_ci_high = wilson_interval(len(concept_valid), len(concept_rows))
            family_table.append(
                {
                    "family": family,
                    "concept": concept,
                    "proposed": len(concept_rows),
                    "valid": len(concept_valid),
                    "valid_rate": round_or_none(len(concept_valid) / len(concept_rows) if concept_rows else None),
                    "valid_rate_ci_low": round_or_none(concept_ci_low),
                    "valid_rate_ci_high": round_or_none(concept_ci_high),
                    "mean_delta_aux": round_or_none(safe_mean(concept_deltas)),
                }
            )
        family_ci_low, family_ci_high = wilson_interval(family_valid, family_prop)
        family_table.append(
            {
                "family": family,
                "concept": "__family_total__",
                "proposed": family_prop,
                "valid": family_valid,
                "valid_rate": round_or_none(family_valid / family_prop if family_prop else None),
                "valid_rate_ci_low": round_or_none(family_ci_low),
                "valid_rate_ci_high": round_or_none(family_ci_high),
                "mean_delta_aux": round_or_none(safe_mean(family_valid_deltas)),
            }
        )

    city_table: list[dict[str, Any]] = []
    for family in FAMILY_ORDER:
        row: dict[str, Any] = {"family": family}
        family_prop_total = 0
        family_valid_total = 0
        for city in city_order:
            scoped = [
                scored_row
                for scored_row in scored_candidate_rows
                if manifest_by_id[Path(scored_row["input_image_path"]).stem]["city"] == city
                and scored_row.get("lever_family") == family
            ]
            prop = len(scoped)
            valid = sum(coerce_bool(item.get("critic_is_valid")) for item in scoped)
            ci_low, ci_high = wilson_interval(valid, prop)
            row[city] = None if prop == 0 else f"{valid}/{prop}"
            row[f"{city}_detail"] = {
                "proposed": prop,
                "valid": valid,
                "valid_rate": round_or_none(valid / prop if prop else None),
                "valid_rate_ci_low": round_or_none(ci_low),
                "valid_rate_ci_high": round_or_none(ci_high),
            }
            family_prop_total += prop
            family_valid_total += valid
        family_ci_low, family_ci_high = wilson_interval(family_valid_total, family_prop_total)
        row["total"] = None if family_prop_total == 0 else f"{family_valid_total}/{family_prop_total}"
        row["total_detail"] = {
            "proposed": family_prop_total,
            "valid": family_valid_total,
            "valid_rate": round_or_none(family_valid_total / family_prop_total if family_prop_total else None),
            "valid_rate_ci_low": round_or_none(family_ci_low),
            "valid_rate_ci_high": round_or_none(family_ci_high),
        }
        city_table.append(row)

    city_totals: dict[str, Any] = {"family": "__city_total__"}
    city_mean_delta: dict[str, Any] = {"family": "__mean_delta__"}
    all_valid_total = 0
    for city in city_order:
        city_rows = [
            scored_row
            for scored_row in scored_candidate_rows
            if manifest_by_id[Path(scored_row["input_image_path"]).stem]["city"] == city
        ]
        city_valid_rows = [row for row in city_rows if coerce_bool(row.get("critic_is_valid"))]
        deltas = [
            float(row["delta_classifier"])
            for row in city_valid_rows
            if row.get("delta_classifier") not in {"", None}
        ]
        city_totals[city] = f"{len(city_valid_rows)}/{len(city_rows)}"
        city_ci_low, city_ci_high = wilson_interval(len(city_valid_rows), len(city_rows))
        city_totals[f"{city}_detail"] = {
            "proposed": len(city_rows),
            "valid": len(city_valid_rows),
            "valid_rate": round_or_none(len(city_valid_rows) / len(city_rows) if city_rows else None),
            "valid_rate_ci_low": round_or_none(city_ci_low),
            "valid_rate_ci_high": round_or_none(city_ci_high),
        }
        city_mean_delta[city] = round_or_none(safe_mean(deltas))
        all_valid_total += len(city_valid_rows)
    city_totals["total"] = f"{len(valid_rows)}/{len(scored_candidate_rows)}"
    total_ci_low, total_ci_high = wilson_interval(len(valid_rows), len(scored_candidate_rows))
    city_totals["total_detail"] = {
        "proposed": len(scored_candidate_rows),
        "valid": len(valid_rows),
        "valid_rate": round_or_none(len(valid_rows) / len(scored_candidate_rows) if scored_candidate_rows else None),
        "valid_rate_ci_low": round_or_none(total_ci_low),
        "valid_rate_ci_high": round_or_none(total_ci_high),
    }
    city_mean_delta["total"] = round_or_none(safe_mean(valid_deltas))
    city_table.extend([city_totals, city_mean_delta])

    mock_or_generation_failures = [
        row
        for row in scored_candidate_rows
        if coerce_bool(row.get("used_mock"))
        or str(row.get("critic_diagnosis", "")).startswith("mock_output=true")
    ]
    invalid_audited = [
        row
        for row in scored_candidate_rows
        if not coerce_bool(row.get("critic_is_valid")) and row not in mock_or_generation_failures
    ]
    failure_table = {
        "generator_failure_before_audit": len(error_rows) + len(mock_or_generation_failures),
        "planner_or_api_failure_rows": len(error_rows),
        "mock_or_generation_failure_rows": len(mock_or_generation_failures),
        "invalid_audited_denominator": len(invalid_audited),
        "implausible_intervention": sum(not coerce_bool(row.get("critic_is_plausible")) for row in invalid_audited),
        "non_local_drift": sum(not coerce_bool(row.get("critic_is_localized")) for row in invalid_audited),
        "unrealistic_rendering": sum(not coerce_bool(row.get("critic_is_realistic")) for row in invalid_audited),
        "same_place_failure": sum(not coerce_bool(row.get("critic_same_place_preserved")) for row in invalid_audited),
        "no_discernible_target_change": sum(not coerce_bool(row.get("critic_edit_attempted")) for row in invalid_audited),
    }

    n_valid_map = {row["image_id"]: maybe_int(row.get("n_valid")) or 0 for row in per_image_rows}
    baseline_xs: list[float] = []
    valid_counts: list[int] = []
    for image_id, baseline in baseline_scores.items():
        baseline_xs.append(float(baseline))
        valid_counts.append(n_valid_map.get(image_id, 0))
    spearman = compute_spearman(baseline_xs, valid_counts)

    ranked_examples: list[dict[str, Any]] = []
    seen_images: set[str] = set()
    for row in sorted(
        (
            row
            for row in valid_rows
            if row.get("delta_classifier") not in {"", None}
        ),
        key=lambda item: float(item["delta_classifier"]),
        reverse=True,
    ):
        image_id = Path(row["input_image_path"]).stem
        if image_id in seen_images:
            continue
        seen_images.add(image_id)
        ranked_examples.append(
            {
                "image_id": image_id,
                "city": manifest_by_id[image_id]["city"],
                "lever_concept": row.get("lever_concept"),
                "lever_family": row.get("lever_family"),
                "scene_support": row.get("scene_support"),
                "delta_aux": round(float(row["delta_classifier"]), 3),
                "baseline_score": round(float(row["baseline_score"]), 3) if row.get("baseline_score") not in {"", None} else None,
                "edited_score": round(float(row["edited_score"]), 3) if row.get("edited_score") not in {"", None} else None,
                "output_image_path": row.get("output_image_path"),
            }
        )
        if len(ranked_examples) == 3:
            break

    return {
        "overall": overall,
        "family_table": family_table,
        "city_table": city_table,
        "failure_table": failure_table,
        "spearman_baseline_vs_valid_count": {
            "n": len(baseline_xs),
            "rho": round_or_none(spearman["rho"], 3),
            "p_value": round_or_none(spearman["p_value"], 3),
        },
        "top_examples": ranked_examples,
    }


def main() -> None:
    args = parse_args()
    payload = summarize(args)
    rendered = json.dumps(payload, indent=2)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
