from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


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

DEFAULT_REPORT_CUTOFFS = [0.0, 0.1, 0.5, 1.0, 2.0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot how many valid levers remain above a delta cutoff, split by family and city."
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
    return parser.parse_args()


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def canonical_city_order(manifest_rows: list[dict[str, str]]) -> list[str]:
    seen = {row["city"] for row in manifest_rows}
    return [city for city in CITY_ORDER if city in seen]


def build_series(
    rows: list[dict[str, str]],
    *,
    group_key: str,
    groups: list[str],
) -> tuple[list[float], dict[str, list[int]]]:
    deltas = sorted({float(row["delta_classifier"]) for row in rows})
    series: dict[str, list[int]] = {group: [] for group in groups}
    for cutoff in deltas:
        kept = [row for row in rows if float(row["delta_classifier"]) >= cutoff]
        counts = defaultdict(int)
        for row in kept:
            counts[row[group_key]] += 1
        for group in groups:
            series[group].append(counts[group])
    return deltas, series


def write_report(
    out_path: str | Path,
    rows: list[dict[str, str]],
    *,
    city_by_image: dict[str, str],
) -> None:
    records: list[dict[str, str | float | int]] = []
    for cutoff in DEFAULT_REPORT_CUTOFFS:
        kept = [row for row in rows if float(row["delta_classifier"]) >= cutoff]

        family_counts = defaultdict(int)
        city_counts = defaultdict(int)
        for row in kept:
            family_counts[row["lever_family"]] += 1
            city_counts[city_by_image[row["image_id"]]] += 1

        for family in FAMILY_ORDER:
            records.append(
                {
                    "group_kind": "family",
                    "group_name": family,
                    "cutoff": cutoff,
                    "count_valid_edits_ge_cutoff": family_counts[family],
                }
            )
        for city in CITY_ORDER:
            if city in city_by_image.values():
                records.append(
                    {
                        "group_kind": "city",
                        "group_name": city,
                        "cutoff": cutoff,
                        "count_valid_edits_ge_cutoff": city_counts[city],
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
                "count_valid_edits_ge_cutoff",
            ],
        )
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    args = parse_args()
    manifest_rows = read_csv(args.manifest_csv)
    scored_rows = read_csv(args.scored_csv)

    city_by_image = {row["image_id"]: row["city"] for row in manifest_rows}
    valid_rows = [row for row in scored_rows if row.get("critic_is_valid") == "True"]
    if not valid_rows:
        raise ValueError("No valid rows found in scored CSV.")

    for row in valid_rows:
        row["city"] = city_by_image[row["image_id"]]

    deltas = [float(row["delta_classifier"]) for row in valid_rows]
    x_min = float(np.floor(min(deltas) * 2.0) / 2.0)
    x_max = float(np.ceil(max(deltas) * 2.0) / 2.0)

    city_order = canonical_city_order(manifest_rows)
    family_cutoffs, family_series = build_series(valid_rows, group_key="lever_family", groups=FAMILY_ORDER)
    city_cutoffs, city_series = build_series(valid_rows, group_key="city", groups=city_order)

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

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), dpi=220, sharey=False)
    family_ax, city_ax = axes

    for family in FAMILY_ORDER:
        family_ax.step(
            family_cutoffs,
            family_series[family],
            where="post",
            label=family,
            linewidth=2.2,
            color=palette[family],
        )
    family_ax.axvline(0.1, color="#6b7280", linestyle="--", linewidth=1.2, alpha=0.8)
    family_ax.set_title("Valid levers above delta cutoff by family")
    family_ax.set_xlabel(r"Delta cutoff $\tau$ (keep edits with $\Delta_{\mathrm{aux}} \geq \tau$)")
    family_ax.set_ylabel("Number of valid edits retained")
    family_ax.set_xlim(x_min, x_max)
    family_ax.grid(True, axis="y", linestyle="--", linewidth=0.8, alpha=0.35)
    family_ax.legend(fontsize=8, frameon=False, loc="upper right")

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
    city_ax.set_title("Valid levers above delta cutoff by city")
    city_ax.set_xlabel(r"Delta cutoff $\tau$ (keep edits with $\Delta_{\mathrm{aux}} \geq \tau$)")
    city_ax.set_xlim(x_min, x_max)
    city_ax.grid(True, axis="y", linestyle="--", linewidth=0.8, alpha=0.35)
    city_ax.legend(fontsize=8, frameon=False, loc="upper right")

    summary_lines = [
        f"n valid edits = {len(valid_rows)}",
        f"threshold 0.1 keeps {sum(delta >= 0.1 for delta in deltas)}",
        f"threshold 0.5 keeps {sum(delta >= 0.5 for delta in deltas)}",
        f"threshold 1.0 keeps {sum(delta >= 1.0 for delta in deltas)}",
    ]
    family_ax.text(
        0.02,
        0.98,
        "\n".join(summary_lines),
        transform=family_ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        bbox={"facecolor": "#fbfaf7", "edgecolor": "#d8d1c4", "boxstyle": "round,pad=0.35"},
    )

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    write_report(args.report_csv, valid_rows, city_by_image=city_by_image)
    print(f"Saved figure -> {out_path}")
    print(f"Saved report -> {args.report_csv}")


if __name__ == "__main__":
    main()
