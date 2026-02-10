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
    openai_model: str = "gpt-5.2"
    use_baseline: bool = True
    baseline_model: str = "black-forest-labs/flux-kontext-max"


@dataclass
class LoggingConfig:
    level: str = "INFO"


@dataclass
class AgentsConfig:
    planner_prompt: str = (
        "You are an urban planner.\n"
        "Given a street-level image and a target percept (e.g., \"safety\", \"wealth\", \"greenery\"), "
        "propose ONE minimal, plausible visual edit that would increase that percept.\n\n"
        "HARD CONSTRAINTS (must follow):\n"
        "1) Single-object edit: choose ONE concrete, physical object/element to modify.\n"
        "2) Localizable: the edit must be achievable by changing pixels only on the target object surface "
        "(or a very small immediately-adjacent area, e.g., mounting bracket/shadow/paint bleed).\n"
        "3) No global changes: do NOT propose edits that imply global relighting, recoloring, weather changes, "
        "camera/viewpoint changes, depth-of-field changes, or overall scene cleanup.\n"
        "4) No multi-object plans: do NOT add/remove multiple objects or create a \"set\" of improvements.\n"
        "5) Minimal magnitude: prefer the smallest change that could plausibly shift perception.\n"
        "6) No text: do not add readable text.\n\n"
        "TARGET OBJECT FORMAT:\n"
        "- target_object must be a short noun phrase (1–4 words), no verbs, no parentheses, "
        "no location descriptions (e.g., not \"streetlight on left\").\n"
        "- Examples: \"streetlight\", \"crosswalk marking\", \"trash bin\", \"storefront shutter\", \"tree canopy\".\n\n"
        "EDIT PLAN FORMAT:\n"
        "- edit_plan must be ONE sentence describing ONE operation.\n"
        "- Use concrete operations like: repair, remove, repaint, brighten, replace, add.\n"
        "- The plan must be specific and directly applicable to the target object.\n"
        "- Avoid embellishment; specify only the exact change required.\n\n"
        "SELF-CHECK (must be true before you answer):\n"
        "- Can the edit be done by editing only the target object pixels (or tiny adjacent area)? YES\n"
        "- Does it avoid global lighting/style changes? YES\n"
        "- Does it avoid adding/removing any other objects? YES\n"
        "- Is it minimal and plausible in the real world? YES\n\n"
        "Return valid JSON with exactly two fields:\n"
        "{\n"
        '  \"edit_plan\": \"<one-sentence minimal edit>\",\n'
        '  \"target_object\": \"<1-4 word noun phrase>\"\n'
        "}\n"
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
        "Target object: {target_object}\n"
        "Edit plan: {edit_plan}\n"
    )

    critic_prompt: str = (
         "Evaluate whether the GENERATED image satisfies the requested edit plan relative to the ORIGINAL image.\n\n"
        "EVALUATION CRITERIA:\n"
        "A) Realism: The edit looks physically plausible and consistent with the scene.\n"
        "B) Minimality: The smallest necessary change was made, localized to the target object.\n"
        "C) No drift: No noticeable changes to camera/viewpoint, global lighting/color, style, or surrounding context.\n\n"
        "FAIL CONDITIONS (examples):\n"
        "- Global style/contrast/saturation shifts.\n"
        "- Any added/removed objects outside the target object.\n"
        "- Camera/viewpoint/geometry changes.\n"
        "- The target object was not edited as requested, or the edit magnitude is too large.\n\n"
        "Return valid JSON:\n"
        "{\n"
        '  "is_realistic": true|false,\n'
        '  "is_minimal_edit": true|false,\n'
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
