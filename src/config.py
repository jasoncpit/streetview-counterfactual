from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict


@dataclass
class ProjectConfig:
    data_root: Path
    raw_dir: Path
    mask_dir: Path
    counterfactual_dir: Path
    eval_dir: Path
    baseline_dir: Path


@dataclass
class WorkflowConfig:
    target_attribute: str = "safety"
    input_dir: Path = Path("data/01_raw")
    max_attempts: int = 3
    openai_model: str = "gpt-5.2"
    concurrency: int = 1
    realism_threshold: float = 0.5
    use_baseline: bool = True
    baseline_model: str = "black-forest-labs/flux-kontext-max"


@dataclass
class LoggingConfig:
    level: str = "INFO"


@dataclass
class AgentsConfig:
    planner_prompt: str = (
        "You are an urban planner. Given a street-level image and a target\n"
        'percept (e.g., "safety", "wealth", "greenery"), propose a visual edit\n'
        'that would increase that percept. Identify a single, concrete object or\n'
        'element to modify (e.g., "streetlight", "tree canopy", "crosswalk marking").\n'
        "The target_object must be a short noun phrase (1-4 words), no verbs,\n"
        "no parentheses, and no location descriptions. Example: \"crosswalk marking\".\n"
        "The edit_plan must be a single, minimal, localized change to that object.\n"
        "Do NOT add other objects, signage, or global scene changes.\n"
        "Avoid embellishments; specify only the exact change needed.\n"
        "Return two fields in JSON:\n"
        "{\n"
        '  "edit_plan": "<what to add/change>",\n'
        '  "target_object": "<specific object to localize>"\n'
        "}\n"
    )
    critic_prompt: str = (
        "You are an image edit critic. Evaluate whether the GENERATED image meets the\n"
        "requirements relative to the ORIGINAL image and the requested plan:\n"
        "(a) faithful to image evidence, (b) minimal + plausible, and (c) no drift of\n"
        "target_object or surrounding context.\n"
        "CRITIC RESPONSIBILITIES:\n"
        "1) Enforce plausibility + minimality: No global restyles, relighting,\n"
        "   camera changes, or broad scene edits. Prefer the smallest number of steps\n"
        "   and minimal edit magnitude localized to target_object.\n"
        "2) Return pass/fail for BOTH realism and minimality.\n"
        "Return valid JSON:\n"
        "{\n"
        '  "is_realistic": true|false,\n'
        '  "is_minimal_edit": true|false,\n'
        '  "notes": "<brief reason and repair guidance for the planner>"\n'
        "}\n"
    )


@dataclass
class ReplicateConfig:
    dino_model: str = "adirik/grounding-dino:latest"
    sam_model: str = (
        "meta/sam-2:fe97b453a6455861e3bac769b441ca1f1086110da7466dbb65cf1eecfd60dc83"
    )
    inpaint_model: str = "black-forest-labs/flux-fill-pro:latest"
    flux_kontext_model: str = "black-forest-labs/flux-kontext-max"
    nano_banana_model: str = "google/nano-banana-pro"
    mock: bool = False
    inpaint_params: Dict[str, Any] = field(default_factory=dict)
    flux_kontext_params: Dict[str, Any] = field(
        default_factory=lambda: {
            "aspect_ratio": "match_input_image",
            "output_format": "jpg",
            "safety_tolerance": 2,
            "prompt_upsampling": False,
        }
    )
    dino_params: Dict[str, Any] = field(
        default_factory=lambda: {"box_threshold": 0.25, "text_threshold": 0.25}
    )
    sam_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolsConfig:
    replicate: ReplicateConfig = field(default_factory=ReplicateConfig)
    download_timeout: int = 360


@dataclass
class AppConfig:
    project: ProjectConfig
    workflow: WorkflowConfig = field(default_factory=WorkflowConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    agents: AgentsConfig = field(default_factory=AgentsConfig)
    tools: ToolsConfig = field(default_factory=ToolsConfig)


def load_config() -> AppConfig:
    repo_root = Path.cwd()
    data_root = repo_root / "data"
    project = ProjectConfig(
        data_root=data_root,
        raw_dir=data_root / "01_raw",
        mask_dir=data_root / "02_masks",
        counterfactual_dir=data_root / "03_counterfactuals",
        eval_dir=data_root / "04_eval_results",
        baseline_dir=data_root / "05_baseline_edits",
    )
    workflow = WorkflowConfig(input_dir=project.raw_dir)
    return AppConfig(project=project, workflow=workflow)
