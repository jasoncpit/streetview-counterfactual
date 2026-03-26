from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console

from src.scoring import (
    THETA_DEFAULT,
    compute_per_image_aux_summary,
    get_last_scoring_device,
    load_baseline_scores,
    load_candidate_rows,
    print_aux_summary,
    score_candidate_rows,
    write_candidate_rows_csv,
    write_per_image_summary_csv,
    write_scatter_svg,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run auxiliary classifier scoring over generated lever CSVs.")
    parser.add_argument("--csv", required=True, help="Input candidate CSV from generate_counterfactual.py")
    parser.add_argument("--attribute", default="safety", help="Perception attribute to score")
    parser.add_argument("--theta", type=float, default=THETA_DEFAULT, help="Exploratory auxiliary threshold")
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "mps", "cuda:0"],
        help="Scoring device. Use cpu to avoid Apple Silicon runtime issues.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Optional output path for the enriched candidate CSV",
    )
    parser.add_argument(
        "--summary-csv",
        type=str,
        default=None,
        help="Optional output path for the per-image auxiliary summary CSV",
    )
    parser.add_argument(
        "--scatter-svg",
        type=str,
        default=None,
        help="Optional output path for the baseline-vs-delta scatter SVG",
    )
    return parser.parse_args()


def default_output_path(input_csv: Path, suffix: str) -> Path:
    return input_csv.with_name(f"{input_csv.stem}{suffix}{input_csv.suffix}")


def default_svg_path(input_csv: Path, suffix: str) -> Path:
    return input_csv.with_name(f"{input_csv.stem}{suffix}.svg")


def main() -> None:
    args = parse_args()
    console = Console()

    input_csv = Path(args.csv)
    output_csv = Path(args.output_csv) if args.output_csv else default_output_path(input_csv, "_scored")
    summary_csv = Path(args.summary_csv) if args.summary_csv else default_output_path(input_csv, "_per_image_aux")
    scatter_svg = Path(args.scatter_svg) if args.scatter_svg else default_svg_path(input_csv, "_baseline_vs_delta")

    rows = load_candidate_rows(input_csv)
    baseline_scores = load_baseline_scores()
    rows = score_candidate_rows(
        rows,
        attribute=args.attribute,
        baseline_scores=baseline_scores,
        theta=args.theta,
        device=args.device,
    )
    summary = compute_per_image_aux_summary(rows, theta=args.theta)

    write_candidate_rows_csv(rows, output_csv)
    write_per_image_summary_csv(summary, summary_csv)
    write_scatter_svg(
        rows,
        scatter_svg,
        theta=args.theta,
        title=f"{args.attribute.title()} baseline vs auxiliary delta",
    )

    console.print(f"[green]Wrote scored candidate CSV:[/green] {output_csv}")
    console.print(f"[green]Wrote per-image summary CSV:[/green] {summary_csv}")
    console.print(f"[green]Wrote scatter SVG:[/green] {scatter_svg}")
    console.print(f"[green]Scoring device:[/green] {get_last_scoring_device()}")
    print_aux_summary(summary)


if __name__ == "__main__":
    main()
