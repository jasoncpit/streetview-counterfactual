from pathlib import Path
from typing import Dict

from src.integrations.replicate_client import ReplicateClient
from src.workflow.state import AgentState


def segment_object_node(
    state: AgentState,
    replicate_client: ReplicateClient,
    mask_dir: Path,
) -> Dict[str, object]:
    """
    Runs a Grounded-SAM-2 style pipeline: Grounding DINO to localize the target object and SAM 2 to produce a mask.
    """
    target_object = state.get("target_object") or "object"
    mask_path = replicate_client.segment_object(
        image_path=state["image_path"],
        prompt=target_object,
        mask_dir=mask_dir,
    )
    return {"mask_path": str(mask_path)}

