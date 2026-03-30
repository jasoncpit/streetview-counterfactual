from __future__ import annotations

import argparse
import csv
import random
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont, ImageOps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build paper figures from reproducible pilot outputs.")
    parser.add_argument(
        "--scored-csv",
        default="data/03_eval_results/specs_paper_n50_scored.csv",
        help="Scored candidate CSV from scripts.run_analysis.",
    )
    parser.add_argument(
        "--summary-csv",
        default="data/03_eval_results/specs_paper_n50_per_image_auxiliary.csv",
        help="Per-image auxiliary summary CSV from scripts.run_analysis.",
    )
    parser.add_argument(
        "--manifest-csv",
        default="data/specs_paper_n50_manifest.csv",
        help="Manifest CSV used to recover per-image city metadata for figure summaries.",
    )
    parser.add_argument(
        "--fig-dir",
        default="paper/figures",
        help="Output directory for generated paper figures.",
    )
    parser.add_argument(
        "--qualitative-mode",
        choices=["top-delta", "random-distinct-family", "image-ids"],
        default="random-distinct-family",
        help="Selection rule for the qualitative panel.",
    )
    parser.add_argument(
        "--qualitative-seed",
        type=int,
        default=8,
        help="Random seed used by qualitative selection modes that sample rows.",
    )
    parser.add_argument(
        "--qualitative-image-ids",
        nargs="*",
        default=[],
        help="Image IDs to visualize in image-ids mode; the best valid edit per image is used.",
    )
    parser.add_argument(
        "--qualitative-lever-concepts",
        nargs="*",
        default=[],
        help="Optional lever concepts aligned with --qualitative-image-ids; when provided, select that valid edit for each image.",
    )
    return parser.parse_args()


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica.ttc",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def build_valid_distribution(summary_rows: list[dict[str, str]], out_path: Path) -> None:
    counts = Counter(int(float(row["n_valid"])) for row in summary_rows)
    xs = sorted(counts)
    ys = [counts[x] for x in xs]
    n_images = len(summary_rows)
    cmap = plt.get_cmap("cividis")
    colors = [cmap(value) for value in [idx / max(1, len(xs) - 1) for idx in range(len(xs))]]

    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=200)
    ax.bar(xs, ys, color=colors, width=0.7)
    ax.set_xlabel("Valid realized levers per image")
    ax.set_ylabel("Number of scenes")
    ax.set_title(f"Distribution of valid realized lever counts on the N={n_images} sample")
    ax.set_xticks(xs)
    ax.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.4)

    mean_coverage = sum(float(row["coverage"]) for row in summary_rows) / len(summary_rows)
    images_with_any_valid = sum(int(float(row["n_valid"])) > 0 for row in summary_rows)
    ax.text(
        0.98,
        0.95,
        f"mean coverage = {mean_coverage:.3f}\nimages with >=1 valid = {images_with_any_valid}/{n_images}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"facecolor": "#fbfaf7", "edgecolor": "#d8d1c4", "boxstyle": "round,pad=0.35"},
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def fit_panel_image(path: str, *, size: tuple[int, int]) -> Image.Image:
    image = Image.open(path).convert("RGB")
    return ImageOps.fit(image, size, method=Image.Resampling.LANCZOS)


def valid_scored_rows(scored_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        row
        for row in scored_rows
        if row.get("critic_is_valid") == "True" and row.get("delta_classifier") not in {"", None}
    ]


def select_qualitative_rows(
    scored_rows: list[dict[str, str]],
    *,
    mode: str,
    seed: int,
    image_ids: list[str],
    lever_concepts: list[str],
) -> list[dict[str, str]]:
    rows = valid_scored_rows(scored_rows)
    if not rows:
        raise ValueError("No valid scored rows available to build the qualitative panel.")

    if mode == "top-delta":
        return sorted(rows, key=lambda row: float(row["delta_classifier"]), reverse=True)[:3]

    if mode == "random-distinct-family":
        by_family: dict[str, list[dict[str, str]]] = {}
        for row in rows:
            by_family.setdefault(row["lever_family"], []).append(row)
        if len(by_family) < 3:
            raise ValueError("Need at least three lever families to sample distinct-family qualitative examples.")

        rng = random.Random(seed)
        selected_families = rng.sample(sorted(by_family), k=3)
        return [rng.choice(by_family[family]) for family in selected_families]

    if mode == "image-ids":
        if not image_ids:
            raise ValueError("image-ids mode requires one or more --qualitative-image-ids values.")
        if lever_concepts and len(lever_concepts) != len(image_ids):
            raise ValueError("--qualitative-lever-concepts must align 1:1 with --qualitative-image-ids.")

        selected: list[dict[str, str]] = []
        for idx, image_id in enumerate(image_ids):
            candidates = [row for row in rows if row["image_id"] == image_id]
            if not candidates:
                raise ValueError(f"No valid scored edits found for image_id={image_id}.")
            if lever_concepts:
                target_concept = lever_concepts[idx]
                candidates = [
                    row for row in candidates if row["lever_concept"] == target_concept
                ]
                if not candidates:
                    raise ValueError(
                        f"No valid scored edits found for image_id={image_id} and lever_concept={target_concept}."
                    )
            selected.append(max(candidates, key=lambda row: float(row["delta_classifier"])))
        return selected

    raise ValueError(f"Unsupported qualitative mode: {mode}")


def build_qualitative_panel(selected_rows: list[dict[str, str]], out_path: Path) -> None:
    ranked = selected_rows

    if not ranked:
        raise ValueError("No valid scored rows available to build the qualitative panel.")

    panel_w = 540
    panel_h = 300
    gutter = 24
    title_h = 54
    row_h = panel_h + title_h + 32
    canvas = Image.new("RGB", (panel_w * 2 + gutter * 3, row_h * len(ranked) + gutter), "#fbfaf7")
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(24)
    body_font = load_font(18)
    small_font = load_font(15)

    for idx, row in enumerate(ranked):
        y0 = gutter + idx * row_h
        title = row["lever_concept"].title()
        delta = float(row["delta_classifier"])
        baseline = float(row["baseline_score"])
        edited = float(row["edited_score"])
        subtitle = f"{row['scene_support']} | delta={delta:+.3f} | {baseline:.2f}->{edited:.2f}"
        draw.text((gutter, y0), f"{idx + 1}. {title}", fill="#1f2937", font=title_font)
        draw.text((gutter, y0 + 28), subtitle, fill="#4b5563", font=small_font)

        original = fit_panel_image(row["input_image_path"], size=(panel_w, panel_h))
        edited_img = fit_panel_image(row["output_image_path"], size=(panel_w, panel_h))
        top = y0 + title_h
        canvas.paste(original, (gutter, top))
        canvas.paste(edited_img, (gutter * 2 + panel_w, top))
        draw.text((gutter, top + panel_h + 8), "Original", fill="#1f2937", font=body_font)
        draw.text((gutter * 2 + panel_w, top + panel_h + 8), "Edited", fill="#1f2937", font=body_font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def build_city_lookup(manifest_rows: list[dict[str, str]]) -> dict[str, str]:
    return {row["image_id"]: row.get("city", "") for row in manifest_rows}


def main() -> None:
    args = parse_args()
    scored_rows = read_csv(args.scored_csv)
    summary_rows = read_csv(args.summary_csv)
    manifest_rows = read_csv(args.manifest_csv)
    fig_dir = Path(args.fig_dir)
    build_valid_distribution(summary_rows, fig_dir / "figure_2_valid_distribution.png")
    selected_rows = select_qualitative_rows(
        scored_rows,
        mode=args.qualitative_mode,
        seed=args.qualitative_seed,
        image_ids=args.qualitative_image_ids,
        lever_concepts=args.qualitative_lever_concepts,
    )
    build_qualitative_panel(selected_rows, fig_dir / "figure_3_qualitative.png")
    city_lookup = build_city_lookup(manifest_rows)
    print(f"qualitative_mode={args.qualitative_mode} seed={args.qualitative_seed}")
    for idx, row in enumerate(selected_rows, start=1):
        city = city_lookup.get(row["image_id"], "Unknown city")
        delta = float(row["delta_classifier"])
        print(
            f"{idx}. {row['lever_family']} | {row['lever_concept']} | "
            f"{city} | image_id={row['image_id']} | delta={delta:+.4f}"
        )
    print(f"Wrote figures to {fig_dir}")


if __name__ == "__main__":
    main()
