from typing import Optional, TypedDict


class AgentState(TypedDict, total=False):
    image_path: str
    target_attribute: str

    # Planner outputs
    edit_plan: Optional[str]
    target_object: Optional[str]

    # Vision tool outputs
    mask_path: Optional[str]
    edited_image_path: Optional[str]

    # Loop control
    attempts: int
    is_realistic: bool
    critic_notes: Optional[str]

