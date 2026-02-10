from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Iterable, Dict, Any

from src.config import AppConfig
from src.integrations.openai_client import Planner
from src.integrations.replicate_client import ReplicateClient
from src.utils.paths import ensure_dir
from src.workflow.graph import build_baseline_workflow
from src.workflow.state import AgentState


def collect_images(input_dir: Path) -> Iterable[Path]:
    patterns = ("*.jpg", "*.jpeg", "*.png", "*.webp")
    for pattern in patterns:
        yield from sorted(input_dir.glob(pattern))


def build_clients(cfg: AppConfig) -> tuple[OpenAIPlanner, ReplicateClient]:
    planner = Planner(
        model=cfg.workflow.openai_model,
        planner_prompt=cfg.agents.planner_prompt,
        critic_prompt=cfg.agents.critic_prompt,
    )
    replicate_client = ReplicateClient()
    return planner, replicate_client


def run_baseline_for_image(
    cfg: AppConfig,
    image_path: Path,
    *,
    baseline_model: str,
    max_attempts: int,
) -> Dict[str, Any]:
    local_cfg = deepcopy(cfg)
    local_cfg.workflow.use_baseline = True
    local_cfg.workflow.baseline_model = baseline_model
    local_cfg.workflow.max_attempts = max_attempts

    input_dir = Path(local_cfg.workflow.input_dir)
    ensure_dir(input_dir)

    planner, replicate_client = build_clients(local_cfg)
    app = build_baseline_workflow(local_cfg, planner, replicate_client)

    state: AgentState = {
        "image_path": str(image_path),
        "target_attribute": local_cfg.workflow.target_attribute,
        "edit_plan": None,
        "target_object": None,
        "mask_path": None,
        "edited_image_path": None,
        "attempts": 0,
        "is_realistic": False,
        "is_minimal_edit": False,
    }

    final_state = app.invoke(state)
    return dict(final_state)
