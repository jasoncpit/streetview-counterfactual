from typing import Dict

from src.integrations.openai_client import Planner
from src.workflow.state import AgentState


def critique_generated_node(
    state: AgentState,
    planner: Planner,
) -> Dict[str, object]:
    """
    Critique the generated image against the original + plan.
    If it fails, return a tightened plan for another attempt.
    """
    image_path = state.get("image_path")
    edited_path = state.get("edited_image_path")
    edit_plan = state.get("edit_plan")
    target_object = state.get("target_object")
    if not image_path or not edited_path or not edit_plan or not target_object:
        raise ValueError("Missing image_path, edited_image_path, edit_plan, or target_object.")
    if state.get("used_mock"):
        return {
            "same_place_preserved": False,
            "is_localized": False,
            "is_realistic": False,
            "is_plausible": False,
            "is_valid": False,
            "critic_notes": "mock_output=true; skipping critique.",
            "attempts": state.get("attempts", 0) + 1,
        }

    critique = planner.critique_generated(
        image_path=image_path,
        edited_image_path=edited_path,
        edit_plan=edit_plan,
        target_object=target_object,
        lever_concept=state.get("lever_concept", "") or "",
        scene_support=state.get("scene_support", "") or "",
        intervention_direction=state.get("intervention_direction", "") or "",
        edit_template=state.get("edit_template", "") or "",
    )
    return {
        "same_place_preserved": critique.same_place_preserved,
        "is_localized": critique.is_localized,
        "is_realistic": critique.is_realistic,
        "is_plausible": critique.is_plausible,
        "is_valid": critique.is_valid,
        "critic_notes": critique.notes,
        "attempts": state.get("attempts", 0) + 1,
    }
