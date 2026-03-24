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
from src.utils.pipeline import collect_images, run_candidate_set_for_image


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
        "--input-path", 
        type=str,
        default=None,
        help="Override input image path. If provided, --input-dir will be ignored.", 
    )
    parser.add_argument(
        "--input-ids",
        type=str,
        default=None,
        help="Optional newline-delimited file of image IDs/stems to process from the input directory.",
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
    parser.add_argument(
        "--candidate-budget",
        type=int,
        default=5,
        help="Maximum number of lever candidates to test per image (default: 5).",
    )
    return parser.parse_args()


def load_input_ids(path: str) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def resolve_images_by_id(input_dir: Path, image_ids: list[str]) -> list[Path]:
    indexed: dict[str, Path] = {}
    for image_path in collect_images(input_dir):
        indexed.setdefault(image_path.stem, image_path)
        indexed.setdefault(image_path.name, image_path)

    missing: list[str] = []
    resolved: list[Path] = []
    for image_id in image_ids:
        image_path = indexed.get(image_id)
        if image_path is None:
            missing.append(image_id)
            continue
        resolved.append(image_path)

    if missing:
        missing_preview = ", ".join(missing[:5])
        raise FileNotFoundError(
            f"Could not resolve {len(missing)} image IDs in {input_dir}: {missing_preview}"
        )
    return resolved


def result_row(image_path: Path, state: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "input_image_path": str(image_path),
        "candidate_id": state.get("candidate_id", ""),
        "target_attribute": state.get("target_attribute", "") or "",
        "lever_concept": state.get("lever_concept", "") or "",
        "scene_support": state.get("scene_support", "") or "",
        "intervention_direction": state.get("intervention_direction", "") or "",
        "edit_template": state.get("edit_template", "") or "",
        "output_image_path": state.get("edited_image_path", "") or "",
        "planner_edit_plan": state.get("edit_plan", "") or "",
        "planner_target_object": state.get("target_object", "") or "",
        "critic_same_place_preserved": bool(state.get("same_place_preserved", False)),
        "critic_is_localized": bool(state.get("is_localized", False)),
        "critic_is_realistic": bool(state.get("is_realistic", False)),
        "critic_is_plausible": bool(state.get("is_plausible", False)),
        "critic_is_valid": bool(state.get("is_valid", False)),
        "attempts_used": state.get("attempts", 0) or 0,
        "used_mock": bool(state.get("used_mock", False)),
        "critic_notes": state.get("critic_notes", "") or "",
    }


def write_csv(rows: list[Dict[str, Any]], csv_path: Path) -> None:
    ensure_dir(csv_path.parent)
    fieldnames = [
        "input_image_path",
        "candidate_id",
        "target_attribute",
        "lever_concept",
        "scene_support",
        "intervention_direction",
        "edit_template",
        "output_image_path",
        "planner_edit_plan",
        "planner_target_object",
        "critic_same_place_preserved",
        "critic_is_localized",
        "critic_is_realistic",
        "critic_is_plausible",
        "critic_is_valid",
        "attempts_used",
        "used_mock",
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
        f"model={args.model} | max_attempts={args.max_attempts} | candidate_budget={args.candidate_budget}"
    )

    results: list[Dict[str, Any]] = []

    if args.input_path:
        image_paths = [Path(args.input_path)]
    elif args.input_ids:
        image_paths = resolve_images_by_id(input_dir, load_input_ids(args.input_ids))
    else:
        image_paths = images
        
    for image_path in tqdm(image_paths, desc="Processing images"):
        try:
            candidate_states = run_candidate_set_for_image(
                cfg,
                image_path,
                baseline_model=args.model,
                max_attempts=args.max_attempts,
                candidate_budget=args.candidate_budget,
            )
            for state in candidate_states:
                results.append(result_row(image_path, state))
        except Exception as err:
            results.append({
                "input_image_path": str(image_path),
                "candidate_id": "",
                "target_attribute": cfg.workflow.target_attribute,
                "lever_concept": "",
                "scene_support": "",
                "intervention_direction": "",
                "edit_template": "",
                "output_image_path": "",
                "planner_edit_plan": "",
                "planner_target_object": "",
                "critic_same_place_preserved": False,
                "critic_is_localized": False,
                "critic_is_realistic": False,
                "critic_is_plausible": False,
                "critic_is_valid": False,
                "attempts_used": 0,
                "used_mock": False,
                "critic_notes": f"ERROR: {err}",
            })

    write_csv(results, csv_path)
    console.print(f"[green]Wrote CSV:[/green] {csv_path}")


if __name__ == "__main__":
    main()
