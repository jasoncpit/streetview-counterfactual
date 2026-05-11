from __future__ import annotations

import argparse
import csv
from collections import defaultdict
import math
from pathlib import Path
import statistics

import matplotlib.pyplot as plt
import numpy as np


FAMILY_ORDER = [
    "Physical Maintenance",
    "Environmental Amenity",
    "Visual Legibility",
    "Mobility Infrastructure",
]

FAMILY_LABELS = {
    "Physical Maintenance": "Maintenance",
    "Environmental Amenity": "Amenity",
    "Visual Legibility": "Legibility",
    "Mobility Infrastructure": "Mobility",
}

CITY_ORDER = [
    "Amsterdam",
    "Abuja",
    "San Francisco",
    "Santiago",
    "Singapore",
]

DEFAULT_REPORT_CUTOFFS = [0.0, 0.1, 0.5, 1.0, 2.0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot the average number of valid levers per scene above a delta cutoff, split by family and city."
    )
    parser.add_argument(
        "scored_csv",
        nargs="?",
        default="data/03_eval_results/specs_paper_n50_scored.csv",
        help="Scored candidate CSV from scripts.run_analysis.",
    )
    parser.add_argument(
        "manifest_csv",
        nargs="?",
        default="data/specs_paper_n50_manifest.csv",
        help="Manifest CSV with image_id to city mapping.",
    )
    parser.add_argument(
        "out_path",
        nargs="?",
        default="paper/figures/figure_5_delta_cutoff_counts.png",
        help="Output image path.",
    )
    parser.add_argument(
        "--report-csv",
        default="data/03_eval_results/specs_paper_n50_delta_cutoff_report.csv",
        help="Optional CSV path for counts at selected cutoff values.",
    )
    parser.add_argument(
        "--summary-csv",
        default="data/03_eval_results/specs_paper_n50_per_image_auxiliary.csv",
        help="Per-image summary CSV from scripts.run_analysis.",
    )
    return parser.parse_args()


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def canonical_city_order(manifest_rows: list[dict[str, str]]) -> list[str]:
    seen = {row["city"] for row in manifest_rows}
    return [city for city in CITY_ORDER if city in seen]


def round_or_none(value: float | None, digits: int = 3) -> float | None:
    if value is None:
        return None
    return round(value, digits)


def safe_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(statistics.mean(values))


def mean_confidence_interval(values: list[float], z: float = 1.96) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    if len(values) == 1:
        return values[0], values[0]
    mean_value = statistics.mean(values)
    stdev = statistics.stdev(values)
    margin = z * (stdev / math.sqrt(len(values)))
    return mean_value - margin, mean_value + margin


def build_series(
    rows: list[dict[str, str]],
    *,
    group_key: str,
    groups: list[str],
    denominator_by_group: dict[str, int],
) -> tuple[list[float], dict[str, list[float]]]:
    deltas = sorted({float(row["delta_classifier"]) for row in rows})
    series: dict[str, list[float]] = {group: [] for group in groups}
    for cutoff in deltas:
        kept = [row for row in rows if float(row["delta_classifier"]) >= cutoff]
        counts = defaultdict(int)
        for row in kept:
            counts[row[group_key]] += 1
        for group in groups:
            denominator = denominator_by_group[group]
            series[group].append(counts[group] / denominator if denominator else 0.0)
    return deltas, series


def write_report(
    out_path: str | Path,
    rows: list[dict[str, str]],
    *,
    city_by_image: dict[str, str],
    total_scenes: int,
    city_scene_counts: dict[str, int],
) -> None:
    records: list[dict[str, str | float | int]] = []
    for cutoff in DEFAULT_REPORT_CUTOFFS:
        kept = [row for row in rows if float(row["delta_classifier"]) >= cutoff]

        family_rows = defaultdict(list)
        city_rows = defaultdict(list)
        for row in kept:
            family_rows[row["lever_family"]].append(row)
            city_rows[city_by_image[row["image_id"]]].append(row)

        for family in FAMILY_ORDER:
            scoped = family_rows[family]
            scoped_deltas = [float(row["delta_classifier"]) for row in scoped]
            delta_ci_low, delta_ci_high = mean_confidence_interval(scoped_deltas)
            records.append(
                {
                    "group_kind": "family",
                    "group_name": family,
                    "cutoff": cutoff,
                    "scene_denominator": total_scenes,
                    "count_valid_edits_ge_cutoff": len(scoped),
                    "avg_valid_levers_per_scene_ge_cutoff": round_or_none(len(scoped) / total_scenes),
                    "mean_delta_aux_ge_cutoff": round_or_none(safe_mean(scoped_deltas)),
                    "mean_delta_aux_ci_low": round_or_none(delta_ci_low),
                    "mean_delta_aux_ci_high": round_or_none(delta_ci_high),
                }
            )
        for city in CITY_ORDER:
            if city in city_by_image.values():
                scoped = city_rows[city]
                scoped_deltas = [float(row["delta_classifier"]) for row in scoped]
                delta_ci_low, delta_ci_high = mean_confidence_interval(scoped_deltas)
                scene_denominator = city_scene_counts[city]
                records.append(
                    {
                        "group_kind": "city",
                        "group_name": city,
                        "cutoff": cutoff,
                        "scene_denominator": scene_denominator,
                        "count_valid_edits_ge_cutoff": len(scoped),
                        "avg_valid_levers_per_scene_ge_cutoff": round_or_none(
                            len(scoped) / scene_denominator if scene_denominator else 0.0
                        ),
                        "mean_delta_aux_ge_cutoff": round_or_none(safe_mean(scoped_deltas)),
                        "mean_delta_aux_ci_low": round_or_none(delta_ci_low),
                        "mean_delta_aux_ci_high": round_or_none(delta_ci_high),
                    }
                )

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "group_kind",
                "group_name",
                "cutoff",
                "scene_denominator",
                "count_valid_edits_ge_cutoff",
                "avg_valid_levers_per_scene_ge_cutoff",
                "mean_delta_aux_ge_cutoff",
                "mean_delta_aux_ci_low",
                "mean_delta_aux_ci_high",
            ],
        )
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    args = parse_args()
    manifest_rows = read_csv(args.manifest_csv)
    scored_rows = read_csv(args.scored_csv)
    summary_rows = read_csv(args.summary_csv)

    city_by_image = {row["image_id"]: row["city"] for row in manifest_rows}
    total_scenes = len({row["image_id"] for row in manifest_rows})
    city_scene_counts = {
        city: len({row["image_id"] for row in manifest_rows if row["city"] == city})
        for city in canonical_city_order(manifest_rows)
    }
    valid_rows = [row for row in scored_rows if row.get("critic_is_valid") == "True"]
    if not valid_rows:
        raise ValueError("No valid rows found in scored CSV.")

    for row in valid_rows:
        row["city"] = city_by_image[row["image_id"]]

    deltas = [float(row["delta_classifier"]) for row in valid_rows]
    mean_delta = safe_mean(deltas)
    mean_delta_ci_low, mean_delta_ci_high = mean_confidence_interval(deltas)
    aux_effective_counts = [
        float(row["n_auxiliary_effective_levers"])
        for row in summary_rows
        if row.get("n_auxiliary_effective_levers") not in {"", None}
    ]
    mean_aux_effective = safe_mean(aux_effective_counts)
    mean_aux_effective_ci_low, mean_aux_effective_ci_high = mean_confidence_interval(aux_effective_counts)
    x_min = float(np.floor(min(deltas) * 2.0) / 2.0)
    x_max = float(np.ceil(max(deltas) * 2.0) / 2.0)

    city_order = canonical_city_order(manifest_rows)
    family_cutoffs, family_series = build_series(
        valid_rows,
        group_key="lever_family",
        groups=FAMILY_ORDER,
        denominator_by_group={family: total_scenes for family in FAMILY_ORDER},
    )
    city_cutoffs, city_series = build_series(
        valid_rows,
        group_key="city",
        groups=city_order,
        denominator_by_group=city_scene_counts,
    )

    palette = {
        "Physical Maintenance": "#4c956c",
        "Environmental Amenity": "#2a9d8f",
        "Visual Legibility": "#bc6c25",
        "Mobility Infrastructure": "#457b9d",
        "Amsterdam": "#355070",
        "Abuja": "#b56576",
        "San Francisco": "#6d597a",
        "Santiago": "#e56b6f",
        "Singapore": "#eaac8b",
    }

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2), dpi=220, sharey=False)
    family_ax, city_ax = axes

    for family in FAMILY_ORDER:
        family_ax.step(
            family_cutoffs,
            family_series[family],
            where="post",
            label=FAMILY_LABELS[family],
            linewidth=2.2,
            color=palette[family],
        )
    family_ax.axvline(0.1, color="#6b7280", linestyle="--", linewidth=1.2, alpha=0.8)
    family_ax.text(0.15, family_ax.get_ylim()[1] * 0.92, r"$\theta_{\mathrm{aux}}=0.1$",
                   fontsize=8.5, color="#6b7280", va="top")
    family_ax.set_title("Average valid levers per scene above cutoff by family")
    family_ax.set_xlabel(r"Delta cutoff $\tau$ (keep edits with $\Delta_{\mathrm{aux}} \geq \tau$)")
    family_ax.set_ylabel("Average valid levers per scene")
    family_ax.set_xlim(x_min, x_max)
    family_ax.grid(True, axis="y", linestyle="--", linewidth=0.8, alpha=0.35)
    family_ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    for city in city_order:
        city_ax.step(
            city_cutoffs,
            city_series[city],
            where="post",
            label=city,
            linewidth=2.2,
            color=palette[city],
        )
    city_ax.axvline(0.1, color="#6b7280", linestyle="--", linewidth=1.2, alpha=0.8)
    city_ax.text(0.15, city_ax.get_ylim()[1] * 0.92, r"$\theta_{\mathrm{aux}}=0.1$",
                 fontsize=8.5, color="#6b7280", va="top")
    city_ax.set_title("Average valid levers per scene above cutoff by city")
    city_ax.set_xlabel(r"Delta cutoff $\tau$ (keep edits with $\Delta_{\mathrm{aux}} \geq \tau$)")
    city_ax.set_xlim(x_min, x_max)
    city_ax.grid(True, axis="y", linestyle="--", linewidth=0.8, alpha=0.35)
    city_ax.legend(loc="upper right", fontsize=9, framealpha=0.9)

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    write_report(
        args.report_csv,
        valid_rows,
        city_by_image=city_by_image,
        total_scenes=total_scenes,
        city_scene_counts=city_scene_counts,
    )
    print(f"Saved figure -> {out_path}")
    print(f"Saved report -> {args.report_csv}")


if __name__ == "__main__":
    main()
