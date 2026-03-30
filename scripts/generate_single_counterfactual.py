from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import load_config
from src.lever_identity import lever_identity_label
from src.scoring import (
    THETA_DEFAULT,
    coerce_bool,
    compute_per_image_aux_summary,
    get_last_scoring_device,
    score_candidate_rows,
    write_candidate_rows_csv,
    write_per_image_summary_csv,
)
from src.utils.logging import configure_logging
from src.utils.paths import ensure_dir, timestamped_path
from src.utils.pipeline import run_candidate_set_for_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate and score a single counterfactual edit for one input image.",
    )
    parser.add_argument("--input-path", required=True, help="Path to the input image.")
    parser.add_argument(
        "--target-attribute",
        default="safety",
        help="Perception attribute to target (default: safety).",
    )
    parser.add_argument(
        "--model",
        default="black-forest-labs/flux-kontext-pro",
        help="Image editing model slug.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=1,
        help="Maximum stochastic attempts for the single candidate lever.",
    )
    parser.add_argument(
        "--candidate-budget",
        type=int,
        default=1,
        help="Planner candidate budget. Keep this at 1 to generate one counterfactual image.",
    )
    parser.add_argument(
        "--theta",
        type=float,
        default=THETA_DEFAULT,
        help="Auxiliary delta threshold used for the printed summary.",
    )
    parser.add_argument(
        "--score-device",
        default="auto",
        choices=["auto", "cpu", "mps", "cuda:0"],
        help="Device for auxiliary scoring.",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Optional output path for the scored candidate CSV.",
    )
    parser.add_argument(
        "--summary-csv",
        default=None,
        help="Optional output path for the per-image summary CSV.",
    )
    return parser.parse_args()


def require_env_vars(console: Console) -> bool:
    required = ("OPENAI_API_KEY", "REPLICATE_API_TOKEN")
    missing = [name for name in required if not os.getenv(name)]
    if not missing:
        return True

    missing_text = "\n".join(f"- {name}" for name in missing)
    console.print(
        Panel.fit(
            f"Missing required environment variables:\n{missing_text}",
            title="Configuration Error",
            border_style="red",
        )
    )
    return False


def build_result_row(image_path: Path, state: dict[str, Any]) -> dict[str, Any]:
    row = {
        "input_image_path": str(image_path),
        "candidate_id": state.get("candidate_id", ""),
        "target_attribute": state.get("target_attribute", "") or "",
        "lever_concept": state.get("lever_concept", "") or "",
        "lever_family": state.get("lever_family", "") or "",
        "scene_support": state.get("scene_support", "") or "",
        "intervention_direction": state.get("intervention_direction", "") or "",
        "edit_template": state.get("edit_template", "") or "",
        "output_image_path": state.get("edited_image_path", "") or "",
        "planner_edit_plan": state.get("edit_plan", "") or "",
        "planner_target_object": state.get("target_object", "") or "",
        "lever_identity_label": "",
        "critic_edit_attempted": bool(state.get("edit_attempted", False)),
        "critic_same_place_preserved": bool(state.get("same_place_preserved", False)),
        "critic_is_localized": bool(state.get("is_localized", False)),
        "critic_is_realistic": bool(state.get("is_realistic", False)),
        "critic_is_plausible": bool(state.get("is_plausible", False)),
        "critic_is_valid": bool(state.get("is_valid", False)),
        "stochastic_attempt_budget": state.get("stochastic_attempt_budget", 0) or 0,
        "stochastic_attempts_used": state.get("attempts", 0) or 0,
        "used_mock": bool(state.get("used_mock", False)),
        "critic_failure_modes": "|".join(str(item) for item in (state.get("critic_failure_modes") or [])),
        "critic_diagnosis": state.get("critic_diagnosis", "") or "",
        "critic_repair_suggestion": state.get("critic_repair_suggestion", "") or "",
        "critic_notes": state.get("critic_notes", "") or "",
    }
    row["lever_identity_label"] = lever_identity_label(row)
    return row


def format_bool(value: Any) -> str:
    return "[green]yes[/green]" if coerce_bool(value) else "[red]no[/red]"


def format_score(value: Any, *, signed: bool = False) -> str:
    if value in (None, ""):
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{number:+.4f}" if signed else f"{number:.4f}"


def default_output_paths(input_path: Path, output_root: Path) -> tuple[Path, Path]:
    stem = f"{input_path.stem}_single_counterfactual"
    csv_path = timestamped_path(output_root, stem, ".csv")
    summary_path = csv_path.with_name(f"{csv_path.stem}_summary.csv")
    return csv_path, summary_path


def select_primary_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def sort_key(row: dict[str, Any]) -> tuple[int, int, float, int]:
        is_valid = 1 if coerce_bool(row.get("critic_is_valid")) else 0
        has_delta = 1 if row.get("delta_classifier") not in (None, "") else 0
        delta = float(row["delta_classifier"]) if has_delta else float("-inf")
        candidate_id = int(row.get("candidate_id") or 0)
        return (is_valid, has_delta, delta, -candidate_id)

    return max(rows, key=sort_key)


def print_overview(
    console: Console,
    *,
    image_path: Path,
    primary_row: dict[str, Any],
    csv_path: Path,
    summary_path: Path,
    theta: float,
) -> None:
    metadata = Table.grid(padding=(0, 2))
    metadata.add_row("Input image", str(image_path))
    metadata.add_row("Edited image", str(primary_row.get("output_image_path") or "n/a"))
    metadata.add_row("Saved scored CSV", str(csv_path))
    metadata.add_row("Saved summary CSV", str(summary_path))
    metadata.add_row("Scoring device", get_last_scoring_device())
    metadata.add_row("Aux threshold", f"{theta:.4f}")
    console.print(Panel.fit(metadata, title="Run Output", border_style="blue"))


def print_candidate_table(console: Console, rows: list[dict[str, Any]], theta: float) -> None:
    table = Table(title="Candidate Results", show_lines=False)
    table.add_column("ID", justify="right")
    table.add_column("Lever", overflow="fold")
    table.add_column("Valid")
    table.add_column("Same place")
    table.add_column("Localized")
    table.add_column("Realistic")
    table.add_column("Plausible")
    table.add_column("Baseline", justify="right")
    table.add_column("Edited", justify="right")
    table.add_column("Delta", justify="right")
    table.add_column(f"> {theta:.2f}")

    for row in rows:
        table.add_row(
            str(row.get("candidate_id") or ""),
            row.get("lever_identity_label") or lever_identity_label(row),
            format_bool(row.get("critic_is_valid")),
            format_bool(row.get("critic_same_place_preserved")),
            format_bool(row.get("critic_is_localized")),
            format_bool(row.get("critic_is_realistic")),
            format_bool(row.get("critic_is_plausible")),
            format_score(row.get("baseline_score")),
            format_score(row.get("edited_score")),
            format_score(row.get("delta_classifier"), signed=True),
            format_bool(row.get("exceeds_auxiliary_threshold")),
        )
    console.print(table)


def print_primary_details(console: Console, row: dict[str, Any]) -> None:
    details = Table.grid(padding=(0, 2))
    details.add_row("Candidate", str(row.get("candidate_id") or "n/a"))
    details.add_row("Lever concept", str(row.get("lever_concept") or "n/a"))
    details.add_row("Lever family", str(row.get("lever_family") or "n/a"))
    details.add_row("Scene support", str(row.get("scene_support") or "n/a"))
    details.add_row("Direction", str(row.get("intervention_direction") or "n/a"))
    details.add_row("Target object", str(row.get("planner_target_object") or "n/a"))
    details.add_row("Edit template", str(row.get("edit_template") or "n/a"))
    details.add_row("Edit plan", str(row.get("planner_edit_plan") or "n/a"))
    details.add_row("Attempts used", str(row.get("stochastic_attempts_used") or 0))
    details.add_row("Mock output", format_bool(row.get("used_mock")))
    details.add_row("Failure modes", str(row.get("critic_failure_modes") or "none"))
    details.add_row("Diagnosis", str(row.get("critic_diagnosis") or "n/a"))
    details.add_row("Repair suggestion", str(row.get("critic_repair_suggestion") or "n/a"))
    details.add_row("Critic notes", str(row.get("critic_notes") or "n/a"))
    console.print(Panel.fit(details, title="Selected Candidate", border_style="green"))


def print_image_summary(console: Console, summary_row: dict[str, Any]) -> None:
    summary = Table.grid(padding=(0, 2))
    summary.add_row("Candidates", str(summary_row.get("n_candidates", 0)))
    summary.add_row("Valid", str(summary_row.get("n_valid", 0)))
    summary.add_row("Scored", str(summary_row.get("n_scored", 0)))
    summary.add_row("Coverage", str(summary_row.get("coverage", 0.0)))
    summary.add_row("Mean delta", str(summary_row.get("mean_delta_classifier", 0.0)))
    summary.add_row("Max delta", str(summary_row.get("max_delta_classifier", 0.0)))
    summary.add_row(
        "Aux-effective levers",
        str(summary_row.get("n_auxiliary_effective_levers", 0)),
    )
    summary.add_row(
        "Effective labels",
        str(summary_row.get("auxiliary_effective_lever_labels") or "none"),
    )
    console.print(Panel.fit(summary, title="Per-Image Summary", border_style="magenta"))


def main() -> int:
    args = parse_args()
    load_dotenv()
    console = Console()

    if not require_env_vars(console):
        return 1

    image_path = Path(args.input_path).expanduser().resolve()
    if not image_path.exists():
        console.print(f"[bold red]Input image not found:[/bold red] {image_path}")
        return 1
    if not image_path.is_file():
        console.print(f"[bold red]Input path is not a file:[/bold red] {image_path}")
        return 1

    cfg = load_config()
    cfg.workflow.target_attribute = args.target_attribute
    configure_logging(cfg.logging.level)

    output_root = Path(cfg.project.eval_dir) / "single_image_runs"
    ensure_dir(output_root)
    csv_path, summary_path = default_output_paths(image_path, output_root)
    if args.output_csv:
        csv_path = Path(args.output_csv)
        summary_path = csv_path.with_name(f"{csv_path.stem}_summary.csv")
    if args.summary_csv:
        summary_path = Path(args.summary_csv)

    console.print(
        f"[bold green]Generating one counterfactual[/bold green] | "
        f"attribute={args.target_attribute} | model={args.model} | "
        f"candidate_budget={args.candidate_budget} | max_attempts={args.max_attempts}"
    )

    try:
        states = run_candidate_set_for_image(
            cfg,
            image_path,
            baseline_model=args.model,
            max_attempts=args.max_attempts,
            candidate_budget=args.candidate_budget,
        )
    except Exception as err:
        console.print(f"[bold red]Generation failed:[/bold red] {err}")
        return 1

    if not states:
        console.print("[yellow]Planner returned no candidate levers for this image.[/yellow]")
        return 1

    rows = [build_result_row(image_path, state) for state in states]
    rows = score_candidate_rows(
        rows,
        attribute=args.target_attribute,
        baseline_scores=None,
        theta=args.theta,
        device=args.score_device,
    )
    summary = compute_per_image_aux_summary(rows, theta=args.theta)

    write_candidate_rows_csv(rows, csv_path)
    write_per_image_summary_csv(summary, summary_path)

    primary_row = select_primary_row(rows)
    summary_row = next(iter(summary.values()))

    print_overview(
        console,
        image_path=image_path,
        primary_row=primary_row,
        csv_path=csv_path,
        summary_path=summary_path,
        theta=args.theta,
    )
    print_candidate_table(console, rows, args.theta)
    print_primary_details(console, primary_row)
    print_image_summary(console, summary_row)

    if coerce_bool(primary_row.get("used_mock")):
        console.print(
            Panel.fit(
                "The edit fell back to a mock output. Check Replicate credentials and model access before using the result.",
                title="Warning",
                border_style="yellow",
            )
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
