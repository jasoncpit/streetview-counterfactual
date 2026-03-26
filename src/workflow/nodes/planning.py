from __future__ import annotations

from typing import Dict

from src.integrations.openai_client import Planner
from src.workflow.state import AgentState


def _sanitize_target_object(raw: str | None) -> str:
    if not raw:
        return "object"
    text = raw.strip()
    for sep in ("(", "[", "{", " - ", ":", ";"):
        if sep in text:
            text = text.split(sep, 1)[0].strip()
    for sep in (" with ", " featuring ", " that ", " to "):
        if sep in text:
            text = text.split(sep, 1)[0].strip()
    text = text.replace('"', "").replace("'", "").strip()
    if text.lower().startswith(("a ", "an ", "the ")):
        text = text.split(" ", 1)[1].strip()
    text = text.rstrip(" .,-")
    words = text.split()
    if len(words) > 4:
        text = " ".join(words[:4])
    return text or "object"


def plan_edit_node(state: AgentState, planner: Planner, target_attribute: str) -> Dict[str, object]:
    """
    Uses OpenAI to propose an edit plan and the target object to localize.
    """
    prior_plan = state.get("edit_plan")
    result = planner.propose_edit(
        image_path=state["image_path"],
        target_attribute=target_attribute,
        prior_plan=prior_plan,
        critic_notes=state.get("critic_notes"),
    )
    return {
        "lever_concept": result.lever_concept,
        "scene_support": result.scene_support,
        "intervention_direction": result.intervention_direction,
        "edit_template": result.edit_template,
        "edit_plan": result.edit_plan,
        "target_object": _sanitize_target_object(result.target_object),
    }
