from typing import Dict

from src.integrations.openai_client import OpenAIPlanner
from src.workflow.state import AgentState


def check_realism_node(
    state: AgentState,
    planner: OpenAIPlanner,
) -> Dict[str, object]:
    """
    LLM-only realism critique (Nano Banana removed). The planner's critique decides pass/fail.
    """
    edited = state.get("edited_image_path")
    if not edited:
        raise ValueError("Missing edited_image_path for criticism node.")

    # No VLM scoring; rely solely on LLM critique.
    realism_score = 1.0
    llm_critique = planner.critique(edited_image_path=edited, notes="vlm_skipped")

    return {
        "is_realistic": llm_critique.is_realistic,
        "critic_notes": f"vlm=skipped; llm={llm_critique.notes}",
    }

