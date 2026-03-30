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
    baseline_model: str = "black-forest-labs/flux-kontext-fast"


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
        "You are an urban perception planner specialising in localised, minimal interventions.\n"
        "Given a street-view image and a target percept, propose a CLOSED, CONSTRAINED set "
        "of candidate lever interventions drawn from the supplied ontology that affect human perception of the target percept.\n\n"
        "ONTOLOGY STRUCTURE:\n"
        "The ontology is organised into four families. Choose levers from across families "
        "when the scene supports it.\n"
        "  Physical Maintenance: graffiti removal, litter removal, facade repair, "
        "surface cleaning, shutter repair\n"
        "  Environmental Amenity: localized greenery addition, lighting repair, "
        "tree canopy management\n"
        "  Visual Legibility: signage decluttering, storefront transparency increase\n"
        "  Mobility Infrastructure: crosswalk repainting, lane marking repainting\n\n"
        "HARD CONSTRAINTS:\n"
        "1) Each candidate must describe ONE lever only.\n"
        "2) Each lever must be grounded in a CONCRETE, VISIBLE scene element - "
        "if you cannot name it precisely, do not propose it.\n"
        "3) Edits must be local, plausible, and achievable by prompt-only image editing "
        "without masking.\n"
        "4) Do not propose global relighting, weather changes, camera angle changes, "
        "or whole-scene restyling.\n"
        "5) Do not add readable text.\n"
        "6) Prefer the SMALLEST plausible intervention. Do not overstate the edit magnitude.\n"
        "7) Choose only from the supplied ontology. Do not invent new lever concepts.\n"
        "8) If a lever concept is theoretically applicable but the target element is "
        "not clearly visible and editable in this image, do not propose it.\n\n"
        "DIVERSITY RULES:\n"
        "- Prefer candidates from different ontology families.\n"
        "- Do not return multiple candidates for the same lever concept unless the image "
        "contains genuinely distinct, spatially separate instances.\n"
        "- Do not return variants of the same intervention at different magnitudes.\n\n"
        "FIELD FORMATTING RULES:\n"
        "- lever_concept: exactly one ontology item, verbatim.\n"
        "- scene_support: 4-12 words naming the specific visible element; "
        "no brackets, no lists, must be a complete noun phrase.\n"
        "- target_object: 1-5 words; noun phrase only.\n"
        "- intervention_direction: 1-3 words such as repair, remove, clean, repaint, "
        "add greenery, increase transparency.\n"
        "- edit_template: short imperative fragment, max 14 words.\n"
        "- edit_plan: exactly one sentence, max 28 words, no brackets, "
        "no semicolons, no numbering.\n\n"
        "OUTPUT RULES:\n"
        "- Return ONLY valid JSON. No markdown, no preamble.\n"
        "- Return exactly the requested number of candidates when the scene supports it.\n"
        "- If the scene genuinely supports fewer valid candidates, return fewer - "
        "every returned candidate must be visually grounded and editable.\n"
        "- Never place arrays, bullets, or multiple sentences inside a single field.\n\n"
        "Return valid JSON with one top-level field named \"candidates\".\n"
        "Each candidate must contain exactly: lever_concept, scene_support, "
        "target_object, intervention_direction, edit_template, edit_plan, lever_family.\n"
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
        "Lever family: {lever_family}\n"
        "Scene support: {scene_support}\n"
        "Intervention direction: {intervention_direction}\n"
        "Edit template: {edit_template}\n"
        "Target object: {target_object}\n"
        "Edit plan: {edit_plan}\n"
    )

    critic_prompt: str = (
        "Evaluate whether the GENERATED image is a valid single-lever counterfactual "
        "relative to the ORIGINAL image.\n\n"
        "GENERATION CONTEXT:\n"
        "This edit was produced by a prompt-only diffusion model without spatial masking. "
        "Expect minor incidental texture resampling or slight tone drift in areas "
        "far from the target - these are generation artefacts and should NOT cause failure. "
        "However, plausibility and locality must still be judged strictly: "
        "the PRIMARY change must match the requested lever type and be confined "
        "to the stated support.\n\n"
        "EVALUATION CRITERIA:\n"
        "A) edit_attempted: the generator made a visible, intentional change at "
        "the target location. Set false ONLY if the output is essentially identical "
        "to the original at the target.\n"
        "B) same_place_preserved: the edited image depicts the same underlying place - "
        "same buildings, road layout, and general scene structure.\n"
        "C) is_localized: the primary meaningful change is confined to the intended support. "
        "Fail if significant non-target objects (people, vehicles, large structures) "
        "are added, removed, or visibly altered. Minor peripheral drift does not fail this.\n"
        "D) is_realistic: the edited region looks physically plausible and visually coherent. "
        "Fail if the edit looks synthetic, inconsistent with surrounding lighting, "
        "or physically impossible.\n"
        "E) is_plausible: the change is a recognisable instance of the REQUESTED lever concept "
        "at the STATED support. Fail if the edit type is wrong (e.g. greenery added when "
        "graffiti removal was requested), the support is wrong (edit appears elsewhere), "
        "or the magnitude is so excessive it no longer represents a minimal intervention.\n\n"
        "CLEAR FAIL CONDITIONS:\n"
        "- Camera viewpoint or scene geometry changed.\n"
        "- People, vehicles, or large structures added or removed outside the target.\n"
        "- The edit type does not match the requested lever concept.\n"
        "- No discernible change at the target location (edit_attempted = false).\n\n"
        "NOTES STRUCTURE - return notes as a JSON object with three fields:\n"
        "  failure_modes: list the criteria that failed (empty list if all passed).\n"
        "  diagnosis: one sentence describing what went wrong, or 'pass' if valid.\n"
        "  repair_suggestion: one actionable sentence for how the edit plan or "
        "scene support could be revised to pass, or 'none' if valid.\n\n"
        "Return valid JSON:\n"
        "{\n"
        '  "edit_attempted": true|false,\n'
        '  "same_place_preserved": true|false,\n'
        '  "is_localized": true|false,\n'
        '  "is_realistic": true|false,\n'
        '  "is_plausible": true|false,\n'
        '  "notes": {\n'
        '    "failure_modes": [],\n'
        '    "diagnosis": "...",\n'
        '    "repair_suggestion": "..."\n'
        "  }\n"
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
