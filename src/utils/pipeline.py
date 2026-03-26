from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Iterable, Dict, Any

from src.config import AppConfig
from src.integrations.openai_client import LeverCandidate, Planner
from src.integrations.replicate_client import ReplicateClient
from src.utils.paths import ensure_dir
from src.workflow.state import AgentState


def collect_images(input_dir: Path) -> Iterable[Path]:
    patterns = ("*.jpg", "*.jpeg", "*.png", "*.webp")
    for pattern in patterns:
        yield from sorted(input_dir.glob(pattern))


def build_clients(cfg: AppConfig) -> tuple[Planner, ReplicateClient]:
    planner = Planner(
        model=cfg.workflow.openai_model,
        planner_prompt=cfg.agents.planner_prompt,
        critic_prompt=cfg.agents.critic_prompt,
    )
    replicate_client = ReplicateClient()
    return planner, replicate_client


def _candidate_to_state(candidate: LeverCandidate) -> AgentState:
    return {
        "lever_concept": candidate.lever_concept,
        "scene_support": candidate.scene_support,
        "intervention_direction": candidate.intervention_direction,
        "edit_template": candidate.edit_template,
        "edit_plan": candidate.edit_plan,
        "target_object": candidate.target_object,
    }


def _run_candidate_attempts(
    cfg: AppConfig,
    planner: Planner,
    replicate_client: ReplicateClient,
    image_path: Path,
    candidate: LeverCandidate,
    *,
    baseline_model: str,
    max_attempts: int,
) -> Dict[str, Any]:
    state: AgentState = {
        "image_path": str(image_path),
        "target_attribute": cfg.workflow.target_attribute,
        "attempts": 0,
        "stochastic_attempt_budget": max_attempts,
        "same_place_preserved": False,
        "is_localized": False,
        "is_realistic": False,
        "is_plausible": False,
        "is_valid": False,
        "critic_notes": "",
        **_candidate_to_state(candidate),
    }

    for attempt in range(1, max_attempts + 1):
        edited_path = replicate_client.image_edit_baseline(
            output_dir=Path(cfg.project.baseline_dir),
            model=baseline_model,
            image_path=str(image_path),
            edit_plan=candidate.edit_plan,
            target_object=candidate.target_object,
            prompt_template=cfg.agents.baseline_edit_prompt,
            lever_concept=candidate.lever_concept,
            scene_support=candidate.scene_support,
            intervention_direction=candidate.intervention_direction,
            edit_template=candidate.edit_template,
        )
        state["edited_image_path"] = str(edited_path) if edited_path else None
        state["used_mock"] = getattr(replicate_client, "last_baseline_used_mock", False)
        state["attempts"] = attempt
        state["stochastic_attempt_index"] = attempt

        if not edited_path or state["used_mock"]:
            state.update(
                {
                    "same_place_preserved": False,
                    "is_localized": False,
                    "is_realistic": False,
                    "is_plausible": False,
                    "is_valid": False,
                    "critic_notes": "mock_output=true; skipping critique.",
                }
            )
            continue

        critique = planner.critique_generated(
            image_path=str(image_path),
            edited_image_path=str(edited_path),
            edit_plan=candidate.edit_plan,
            target_object=candidate.target_object,
            lever_concept=candidate.lever_concept,
            scene_support=candidate.scene_support,
            intervention_direction=candidate.intervention_direction,
            edit_template=candidate.edit_template,
        )
        state.update(
            {
                "same_place_preserved": critique.same_place_preserved,
                "is_localized": critique.is_localized,
                "is_realistic": critique.is_realistic,
                "is_plausible": critique.is_plausible,
                "is_valid": critique.is_valid,
                "critic_notes": critique.notes,
            }
        )
        if critique.is_valid:
            break

    return dict(state)


def run_candidate_set_for_image(
    cfg: AppConfig,
    image_path: Path,
    *,
    baseline_model: str,
    max_attempts: int,
    candidate_budget: int,
) -> list[Dict[str, Any]]:
    local_cfg = deepcopy(cfg)
    local_cfg.workflow.use_baseline = True
    local_cfg.workflow.baseline_model = baseline_model
    local_cfg.workflow.max_attempts = max_attempts
    local_cfg.workflow.candidate_set_size = candidate_budget

    input_dir = Path(local_cfg.workflow.input_dir)
    ensure_dir(input_dir)

    planner, replicate_client = build_clients(local_cfg)
    candidate_result = planner.propose_lever_candidates(
        image_path=str(image_path),
        target_attribute=local_cfg.workflow.target_attribute,
        lever_ontology=local_cfg.agents.lever_ontology,
        candidate_budget=candidate_budget,
    )

    rows: list[Dict[str, Any]] = []
    for idx, candidate in enumerate(candidate_result.candidates, start=1):
        candidate_state = _run_candidate_attempts(
            local_cfg,
            planner,
            replicate_client,
            image_path,
            candidate,
            baseline_model=baseline_model,
            max_attempts=max_attempts,
        )
        candidate_state["candidate_id"] = idx
        rows.append(candidate_state)
    return rows
