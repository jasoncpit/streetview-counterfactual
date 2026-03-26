from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict


@dataclass
class ProjectConfig:
    data_root: Path
    raw_dir: Path
    eval_dir: Path
    baseline_dir: Path


@dataclass
class WorkflowConfig:
    target_attribute: str = "safety"
    input_dir: Path = Path("data/01_raw")
    max_attempts: int = 3
    candidate_set_size: int = 5
    openai_model: str = "gpt-5.2"
    use_baseline: bool = True
    baseline_model: str = "black-forest-labs/flux-kontext-max"


@dataclass
class LoggingConfig:
    level: str = "INFO"


@dataclass
class AgentsConfig:
    lever_ontology: tuple[str, ...] = (
        "graffiti removal",
        "litter removal",
        "facade repair",
        "storefront transparency increase",
        "localized greenery addition",
        "lighting repair",
        "signage decluttering",
        "crosswalk repainting",
        "lane marking repainting",
        "surface cleaning",
        "shutter repair",
        "tree canopy management",
    )
    planner_prompt: str = (
        "You are an urban perception planner.\n"
        "Given a street-view image and a target percept, propose a CLOSED, CONSTRAINED set of candidate lever interventions.\n\n"
        "HARD CONSTRAINTS:\n"
        "1) Each candidate must describe ONE lever only.\n"
        "2) Each lever must be grounded in a concrete visible scene element.\n"
        "3) Edits must be local, plausible, and prompt-only friendly.\n"
        "4) Do not propose global relighting, weather changes, camera changes, or whole-scene cleanup.\n"
        "5) Do not add readable text.\n"
        "6) Prefer the smallest plausible intervention that could shift the target percept.\n"
        "7) Choose only from the supplied ontology.\n\n"
        "DIVERSITY RULES:\n"
        "- Return distinct candidates, not variants of the same lever.\n"
        "- Use different lever concepts whenever the scene supports it.\n"
        "- Do not return multiple lane-marking variants unless the image truly contains no other credible lever.\n\n"
        "FIELD FORMATTING RULES:\n"
        "- lever_concept: exactly one ontology item.\n"
        "- scene_support: 4-12 words naming the visible support; no brackets, no lists, no sentence fragments ending in articles/prepositions.\n"
        "- target_object: 1-5 words; noun phrase only.\n"
        "- intervention_direction: 1-3 words such as repair, remove, clean, repaint, brighten, add greenery, increase transparency.\n"
        "- edit_template: short imperative fragment, max 14 words.\n"
        "- edit_plan: exactly one sentence, max 28 words, no brackets, no numbering, no semicolons.\n\n"
        "OUTPUT RULES:\n"
        "- Return ONLY valid JSON. No markdown.\n"
        "- Return exactly the requested number of candidates when possible.\n"
        "- If the scene supports fewer candidates, return fewer, but every returned candidate must be valid and grounded.\n"
        "- Never place arrays, bullets, or multiple sentences inside a single field.\n\n"
        "Return valid JSON with one top-level field named \"candidates\".\n"
        "Each candidate must contain exactly these fields:\n"
        "- lever_concept\n"
        "- scene_support\n"
        "- target_object\n"
        "- intervention_direction\n"
        "- edit_template\n"
        "- edit_plan\n"
    )

    baseline_edit_prompt: str = (
        "Use the PROVIDED image as the base. Do NOT generate a new scene.\n"
        "Preserve the exact camera viewpoint, geometry, and scene layout.\n\n"
        "ALLOWED CHANGE:\n"
        "- Only modify the target object described below, and only as much as required by the plan.\n"
        "- If repainting/retexturing, keep shape and placement identical; adjust only the target object's "
        "surface appearance.\n\n"
        "FORBIDDEN:\n"
        "- No global restyling, relighting, recoloring, or contrast/saturation shifts.\n"
        "- Do not add/remove any other objects (people, vehicles, signs, markings, street furniture, plants, etc.).\n"
        "- Do not introduce readable text.\n"
        "- Do not change the background, sky, buildings, road texture, or any surrounding context.\n\n"
        "QUALITY BAR:\n"
        "- The edit should look realistic and physically plausible.\n"
        "- Outside the target object area, the image should appear unchanged to a human observer.\n\n"
        "Lever concept: {lever_concept}\n"
        "Scene support: {scene_support}\n"
        "Intervention direction: {intervention_direction}\n"
        "Edit template: {edit_template}\n"
        "Target object: {target_object}\n"
        "Edit plan: {edit_plan}\n"
    )

    critic_prompt: str = (
         "Evaluate whether the GENERATED image satisfies the requested lever intervention relative to the ORIGINAL image.\n\n"
        "EVALUATION CRITERIA:\n"
        "A) same_place_preserved: the edited image still depicts the same underlying place.\n"
        "B) is_localized: changes remain substantially confined to the intended support/target object.\n"
        "C) is_realistic: the edit looks physically plausible and visually coherent.\n"
        "D) is_plausible: the change is a credible instance of the requested lever intervention.\n\n"
        "FAIL CONDITIONS (examples):\n"
        "- Global style/contrast/saturation shifts.\n"
        "- Any added/removed objects outside the target object.\n"
        "- Camera/viewpoint/geometry changes.\n"
        "- The target object was not edited as requested, or the edit magnitude is too large.\n"
        "- The requested lever concept is replaced by a different kind of change.\n\n"
        "Return valid JSON:\n"
        "{\n"
        '  "same_place_preserved": true|false,\n'
        '  "is_localized": true|false,\n'
        '  "is_realistic": true|false,\n'
        '  "is_plausible": true|false,\n'
        '  "notes": "<brief reason and repair guidance for the planner>"\n'
        "}\n"
    )



@dataclass
class AppConfig:
    project: ProjectConfig
    workflow: WorkflowConfig = field(default_factory=WorkflowConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    agents: AgentsConfig = field(default_factory=AgentsConfig)


def load_config() -> AppConfig:
    repo_root = Path.cwd()
    data_root = repo_root / "data"
    project = ProjectConfig(
        data_root=data_root,
        raw_dir=data_root / "01_raw",
        baseline_dir=data_root / "02_counterfactual",
        eval_dir=data_root / "03_eval_results",
    )
    workflow = WorkflowConfig(input_dir=project.raw_dir)
    return AppConfig(project=project, workflow=workflow)
