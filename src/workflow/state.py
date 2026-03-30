from typing import Optional, TypedDict


class AgentState(TypedDict, total=False):
    image_path: str
    target_attribute: str

    # Planner outputs
    lever_concept: Optional[str]
    lever_family: Optional[str]
    scene_support: Optional[str]
    intervention_direction: Optional[str]
    edit_template: Optional[str]
    edit_plan: Optional[str]
    target_object: Optional[str]

    # Vision tool outputs
    mask_path: Optional[str]
    edited_image_path: Optional[str]

    # Loop control
    attempts: int
    stochastic_attempt_budget: int
    stochastic_attempt_index: int
    edit_attempted: Optional[bool]
    same_place_preserved: Optional[bool]
    is_localized: Optional[bool]
    is_realistic: bool
    is_plausible: Optional[bool]
    is_valid: Optional[bool]
    critic_notes: Optional[str]
    critic_failure_modes: Optional[list[str]]
    critic_diagnosis: Optional[str]
    critic_repair_suggestion: Optional[str]
    used_mock: Optional[bool]
