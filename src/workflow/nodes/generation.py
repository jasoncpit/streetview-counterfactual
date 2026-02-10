from pathlib import Path
from typing import Dict

from src.integrations.replicate_client import ReplicateClient
from src.workflow.state import AgentState


def inpaint_node(
    state: AgentState,
    replicate_client: ReplicateClient,
    output_dir: Path,
) -> Dict[str, object]:
    """
    Performs inpainting on the masked region using the edit plan as a prompt.
    """
    image_path = state["image_path"]
    mask_path = state.get("mask_path")
    prompt = state.get("edit_plan") or "Improve the scene."

    if not mask_path:
        # Fail-fast to avoid confusing downstream nodes
        raise ValueError("Missing mask_path in state before generation.")

    edited_image = replicate_client.inpaint(
        image_path=image_path,
        mask_path=mask_path,
        prompt=prompt,
        output_dir=output_dir,
    )
    return {"edited_image_path": str(edited_image)}
