from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv
from hydra import main
from omegaconf import DictConfig
from rich.console import Console

from src.integrations.openai_client import OpenAIPlanner
from src.integrations.replicate_client import ReplicateClient, ReplicateModels
from src.utils.logging import configure_logging
from src.utils.paths import ensure_dir
from src.workflow.graph import build_workflow
from src.workflow.state import AgentState


def collect_images(input_dir: Path) -> Iterable[Path]:
    patterns = ("*.jpg", "*.jpeg", "*.png", "*.webp")
    for pattern in patterns:
        yield from sorted(input_dir.glob(pattern))


@main(config_path="../configs", config_name="main", version_base=None)
def run(cfg: DictConfig) -> None:
    load_dotenv()
    configure_logging(cfg.logging.level)
    console = Console()

    input_dir = Path(cfg.workflow.input_dir)
    ensure_dir(input_dir)

    planner = OpenAIPlanner(
        model=cfg.workflow.openai_model,
        planner_prompt=cfg.agents.planner_prompt,
        critic_prompt=cfg.agents.critic_prompt,
    )
    replicate_models = ReplicateModels.from_config(cfg.tools.replicate)
    replicate_client = ReplicateClient(models=replicate_models)

    app = build_workflow(cfg, planner, replicate_client)
    images = list(collect_images(input_dir))

    if not images:
        console.print(f"[yellow]No images found in {input_dir}[/yellow]")
        return

    console.print(
        f"[bold green]Launching pipeline[/bold green] for {len(images)} image(s) | "
        f"target_attribute={cfg.workflow.target_attribute}"
    )

    for image_path in images:
        state: AgentState = {
            "image_path": str(image_path),
            "target_attribute": cfg.workflow.target_attribute,
            "edit_plan": None,
            "target_object": None,
            "mask_path": None,
            "edited_image_path": None,
            "attempts": 0,
            "is_realistic": False,
        }

        console.rule(title=f"[cyan]Processing[/cyan] {image_path.name}")
        for event in app.stream(state, stream_mode="updates"):
            if isinstance(event, tuple) and len(event) == 2:
                node, payload = event
            elif isinstance(event, dict):
                node = event.get("node") or event.get("name") or "event"
                payload = event.get("output") or event.get("value") or event
            else:
                node, payload = "event", event
            console.print(f"[blue]{node}[/blue]: {payload}")

        console.print(f"[green]Done:[/green] {image_path.name}")


if __name__ == "__main__":
    run()

