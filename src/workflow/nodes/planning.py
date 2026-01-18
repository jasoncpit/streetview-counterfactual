from typing import Dict

from src.integrations.openai_client import OpenAIPlanner
from src.workflow.state import AgentState


def plan_edit_node(state: AgentState, planner: OpenAIPlanner, target_attribute: str) -> Dict[str, object]:
    """
    Uses OpenAI to propose an edit plan and the target object to localize.
    """
    prior_plan = state.get("edit_plan")
    result = planner.propose_edit(
        image_path=state["image_path"],
        target_attribute=target_attribute,
        prior_plan=prior_plan,
    )
    attempts = state.get("attempts", 0) + 1
    return {
        "edit_plan": result.edit_plan,
        "target_object": result.target_object,
        "attempts": attempts,
    }

