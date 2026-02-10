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
            "is_realistic": False,
            "is_minimal_edit": False,
            "critic_notes": "mock_output=true; skipping critique.",
            "attempts": state.get("attempts", 0) + 1,
        }

    critique = planner.critique_generated(
        image_path=image_path,
        edited_image_path=edited_path,
        edit_plan=edit_plan,
        target_object=target_object,
    )
    return {
        "is_realistic": critique.is_realistic,
        "is_minimal_edit": critique.is_minimal_edit,
        "critic_notes": critique.notes,
        "attempts": state.get("attempts", 0) + 1,
    }
