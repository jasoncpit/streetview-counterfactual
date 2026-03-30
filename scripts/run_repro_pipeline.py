from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a reproducible subset of the paper pipeline end-to-end.",
    )
    parser.add_argument("--prefix", default="specs_repro_n20", help="Run prefix used for all outputs.")
    parser.add_argument("--city", default="all", help="City filter for subset preparation.")
    parser.add_argument("--attribute", default="safe", help="Attribute used for subset preparation and scoring.")
    parser.add_argument("--target-attribute", default="safety", help="Target attribute used for generation.")
    parser.add_argument("--n-total", type=int, default=20, help="Number of images to prepare.")
    parser.add_argument("--score-bins", type=int, default=5, help="Score bins for subset stratification.")
    parser.add_argument(
        "--complexity-bins",
        type=int,
        default=2,
        help="Visual complexity bins for subset stratification.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for subset selection.")
    parser.add_argument("--candidate-budget", type=int, default=3, help="Lever candidates per image.")
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=1,
        help="Bounded stochastic attempts per candidate lever.",
    )
    parser.add_argument(
        "--model",
        default="black-forest-labs/flux-kontext-pro",
        help="Image editing model slug.",
    )
    parser.add_argument(
        "--score-device",
        default="auto",
        choices=["auto", "cpu", "mps", "cuda:0"],
        help="Device for auxiliary scoring.",
    )
    parser.add_argument(
        "--skip-analysis",
        action="store_true",
        help="Skip auxiliary scoring and only run subset prep + generation.",
    )
    parser.add_argument(
        "--export-human-eval",
        action="store_true",
        help="Also export accepted edits for optional E2 human evaluation.",
    )
    return parser.parse_args()


def run(cmd: list[str], *, cwd: Path) -> None:
    print(f"\n$ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=cwd, check=True)


def write_config(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    prefix = args.prefix

    subset_dir = Path("data/01_raw") / prefix
    ids_file = Path("data") / f"{prefix}_ids.txt"
    manifest_path = Path("data") / f"{prefix}_manifest.csv"
    summary_path = Path("data") / f"{prefix}_summary.json"
    score_cache = Path("data") / f"{prefix}_scores.json"
    candidate_csv = Path("data/03_eval_results") / f"{prefix}.csv"
    scored_csv = Path("data/03_eval_results") / f"{prefix}_scored.csv"
    summary_csv = Path("data/03_eval_results") / f"{prefix}_per_image_auxiliary.csv"
    scatter_svg = Path("data/03_eval_results") / f"{prefix}_baseline_vs_aux_delta.svg"
    config_path = Path("data") / f"{prefix}_config.json"

    config = {
        "prefix": prefix,
        "city": args.city,
        "attribute": args.attribute,
        "target_attribute": args.target_attribute,
        "n_total": args.n_total,
        "score_bins": args.score_bins,
        "complexity_bins": args.complexity_bins,
        "seed": args.seed,
        "candidate_budget": args.candidate_budget,
        "max_attempts": args.max_attempts,
        "model": args.model,
        "score_device": args.score_device,
        "export_human_eval": args.export_human_eval,
        "subset_dir": str(subset_dir),
        "ids_file": str(ids_file),
        "manifest_path": str(manifest_path),
        "summary_path": str(summary_path),
        "score_cache": str(score_cache),
        "candidate_csv": str(candidate_csv),
        "scored_csv": str(scored_csv),
        "summary_csv": str(summary_csv),
        "scatter_svg": str(scatter_svg),
    }
    write_config(root / config_path, config)

    py = sys.executable

    run(
        [
            py,
            "scripts/prepare_specs.py",
            "--city",
            args.city,
            "--attribute",
            args.attribute,
            "--n-total",
            str(args.n_total),
            "--score-bins",
            str(args.score_bins),
            "--complexity-bins",
            str(args.complexity_bins),
            "--seed",
            str(args.seed),
            "--output-dir",
            str(subset_dir),
            "--ids-file",
            str(ids_file),
            "--manifest-path",
            str(manifest_path),
            "--summary-path",
            str(summary_path),
            "--score-cache-path",
            str(score_cache),
            "--clean-output-dir",
        ],
        cwd=root,
    )

    run(
        [
            py,
            "-m",
            "scripts.generate_counterfactual",
            "--input-dir",
            str(subset_dir),
            "--input-ids",
            str(ids_file),
            "--candidate-budget",
            str(args.candidate_budget),
            "--max-attempts",
            str(args.max_attempts),
            "--target-attribute",
            args.target_attribute,
            "--model",
            args.model,
            "--csv-path",
            str(candidate_csv),
            "--resume",
        ],
        cwd=root,
    )

    if not args.skip_analysis:
        run(
            [
                "uv",
                "run",
                "--isolated",
                "--no-project",
                "--python",
                "3.12",
                "--with",
                "torch",
                "--with",
                "torchvision",
                "--with",
                "huggingface_hub",
                "--with",
                "pillow",
                "--with",
                "rich",
                "python",
                "-m",
                "scripts.run_analysis",
                "--csv",
                str(candidate_csv),
                "--attribute",
                args.target_attribute,
                "--score-cache",
                str(score_cache),
                "--device",
                args.score_device,
                "--output-csv",
                str(scored_csv),
                "--summary-csv",
                str(summary_csv),
                "--scatter-svg",
                str(scatter_svg),
            ],
            cwd=root,
        )

    if args.export_human_eval:
        run(
            [
                py,
                "-m",
                "scripts.export_human_eval_e2",
                "--csv",
                str(candidate_csv),
            ],
            cwd=root,
        )

    print("\nReproducible pipeline run complete.")
    print(f"Config:        {config_path}")
    print(f"Subset dir:    {subset_dir}")
    print(f"Candidate CSV: {candidate_csv}")
    if not args.skip_analysis:
        print(f"Scored CSV:    {scored_csv}")
        print(f"Summary CSV:   {summary_csv}")
        print(f"Scatter SVG:   {scatter_svg}")


if __name__ == "__main__":
    main()
