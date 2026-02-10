from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv
from rich.console import Console
from tqdm import tqdm

from src.config import load_config
from src.utils.logging import configure_logging
from src.utils.paths import ensure_dir
from src.utils.pipeline import collect_images, run_baseline_for_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate counterfactuals for all images in 01_raw.")
    parser.add_argument(
        "--model",
        default="black-forest-labs/flux-kontext-max",
        help="Baseline edit model slug (default: google/nano-banana-pro).",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=1,
        help="Maximum number of plan/edit attempts per image (default: 3).",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="Override input directory (default: data/01_raw).",
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        default=None,
        help="Optional output CSV path.",
    )
    parser.add_argument(
        "--target-attribute",
        type=str,
        default="safety",
        help="Target attribute to generate counterfactuals for (default: safety).",
    )
    return parser.parse_args()


def result_row(image_path: Path, state: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "input_image_path": str(image_path),
        "output_image_path": state.get("edited_image_path", "") or "",
        "planner_edit_plan": state.get("edit_plan", "") or "",
        "planner_target_object": state.get("target_object", "") or "",
        "critic_is_realistic": bool(state.get("is_realistic", False)),
        "critic_is_minimal_edit": bool(state.get("is_minimal_edit", False)),
        "critic_notes": state.get("critic_notes", "") or "",
    }


def write_csv(rows: list[Dict[str, Any]], csv_path: Path) -> None:
    ensure_dir(csv_path.parent)
    fieldnames = [
        "input_image_path",
        "output_image_path",
        "planner_edit_plan",
        "planner_target_object",
        "critic_is_realistic",
        "critic_is_minimal_edit",
        "critic_notes",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    cfg = load_config()
    load_dotenv()
    configure_logging(cfg.logging.level)
    console = Console()

    if args.input_dir:
        cfg.workflow.input_dir = args.input_dir
    if args.target_attribute:
        cfg.workflow.target_attribute = args.target_attribute
    input_dir = Path(cfg.workflow.input_dir)
    ensure_dir(input_dir)
    images = list(collect_images(input_dir))
    if not images:
        console.print(f"[yellow]No images found in {input_dir}[/yellow]")
        return

    if args.csv_path:
        csv_path = Path(args.csv_path)
    else:
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        csv_path = Path(cfg.project.eval_dir) / f"counterfactual_results_{timestamp}.csv"

    console.print(
        f"[bold green]Generating counterfactuals[/bold green] | images={len(images)} | "
        f"model={args.model} | max_attempts={args.max_attempts}"
    )

    results: list[Dict[str, Any]] = []

    for image_path in tqdm(images, desc="Processing images"):
        try:
            final_state = run_baseline_for_image(
                cfg,
                image_path,
                baseline_model=args.model,
                max_attempts=args.max_attempts,
            )
            row = result_row(image_path, final_state)
        except Exception as err:
            row = {
                "input_image_path": str(image_path),
                "output_image_path": "",
                "planner_edit_plan": "",
                "planner_target_object": "",
                "critic_is_realistic": False,
                "critic_is_minimal_edit": False,
                "critic_notes": f"ERROR: {err}",
            }
        results.append(row)

    write_csv(results, csv_path)
    console.print(f"[green]Wrote CSV:[/green] {csv_path}")


if __name__ == "__main__":
    main()
