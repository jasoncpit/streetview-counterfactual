from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console

from src.config import load_config
from src.utils.logging import configure_logging
from src.utils.paths import ensure_dir
from src.utils.pipeline import collect_images, run_candidate_set_for_image


def run() -> None:
    cfg = load_config()
    load_dotenv()
    configure_logging(cfg.logging.level)
    console = Console()

    input_dir = Path(cfg.workflow.input_dir)
    ensure_dir(input_dir)
    images = list(collect_images(input_dir))

    if not images:
        console.print(f"[yellow]No images found in {input_dir}[/yellow]")
        return

    console.print(
        f"[bold green]Launching candidate pipeline[/bold green] for {len(images)} image(s) | "
        f"target_attribute={cfg.workflow.target_attribute}"
    )

    for image_path in images:
        console.rule(title=f"[cyan]Processing[/cyan] {image_path.name}")
        rows = run_candidate_set_for_image(
            cfg,
            image_path,
            baseline_model=cfg.workflow.baseline_model,
            max_attempts=cfg.workflow.max_attempts,
            candidate_budget=min(cfg.workflow.candidate_set_size, 3),
        )
        for row in rows:
            console.print(
                {
                    "candidate_id": row.get("candidate_id"),
                    "lever_concept": row.get("lever_concept"),
                    "scene_support": row.get("scene_support"),
                    "is_valid": row.get("is_valid"),
                    "critic_notes": row.get("critic_notes"),
                }
            )


if __name__ == "__main__":
    run()
