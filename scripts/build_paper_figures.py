from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont, ImageOps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build paper figures from reproducible pilot outputs.")
    parser.add_argument(
        "--scored-csv",
        default="data/03_eval_results/specs_repro_n20_scored.csv",
        help="Scored candidate CSV from scripts.run_analysis.",
    )
    parser.add_argument(
        "--summary-csv",
        default="data/03_eval_results/specs_repro_n20_per_image_auxiliary.csv",
        help="Per-image auxiliary summary CSV from scripts.run_analysis.",
    )
    parser.add_argument(
        "--fig-dir",
        default="paper/figures",
        help="Output directory for generated paper figures.",
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

    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=200)
    ax.bar(xs, ys, color=["#d9d2c5", "#7aa37a", "#4f7c78", "#284b63"][: len(xs)], width=0.7)
    ax.set_xlabel("Valid realized levers per image")
    ax.set_ylabel("Number of scenes")
    ax.set_title("Distribution of valid realized lever counts on the N=20 pilot")
    ax.set_xticks(xs)
    ax.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.4)

    mean_coverage = sum(float(row["coverage"]) for row in summary_rows) / len(summary_rows)
    images_with_any_valid = sum(int(float(row["n_valid"])) > 0 for row in summary_rows)
    ax.text(
        0.98,
        0.95,
        f"mean coverage = {mean_coverage:.3f}\nimages with >=1 valid = {images_with_any_valid}/{len(summary_rows)}",
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


def build_qualitative_panel(scored_rows: list[dict[str, str]], out_path: Path) -> None:
    ranked: list[dict[str, str]] = []
    seen_images: set[str] = set()
    for row in sorted(
        (
            row
            for row in scored_rows
            if row.get("critic_is_valid") == "True" and row.get("delta_classifier") not in {"", None}
        ),
        key=lambda row: float(row["delta_classifier"]),
        reverse=True,
    ):
        image_id = row["image_id"]
        if image_id in seen_images:
            continue
        ranked.append(row)
        seen_images.add(image_id)
        if len(ranked) == 3:
            break

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


def main() -> None:
    args = parse_args()
    scored_rows = read_csv(args.scored_csv)
    summary_rows = read_csv(args.summary_csv)
    fig_dir = Path(args.fig_dir)
    build_valid_distribution(summary_rows, fig_dir / "figure_2_valid_distribution.png")
    build_qualitative_panel(scored_rows, fig_dir / "figure_3_qualitative.png")
    print(f"Wrote figures to {fig_dir}")


if __name__ == "__main__":
    main()
