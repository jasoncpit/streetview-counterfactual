import base64
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

from openai import OpenAI


@dataclass
class PlanResult:
    edit_plan: str
    target_object: str


@dataclass
class GeneratedCritiqueResult:
    is_realistic: bool
    is_minimal_edit: bool
    notes: str


class OpenAIPlanner:
    """
    Lightweight wrapper over OpenAI Chat Completions for planning and realism checks.
    """

    def __init__(self, model: str, planner_prompt: str, critic_prompt: str) -> None:
        self.client = OpenAI()
        self.model = model
        self.planner_prompt = planner_prompt.strip()
        self.critic_prompt = critic_prompt.strip()

    def _image_to_data_url(self, image_path: str) -> str:
        path = Path(image_path)
        suffix = path.suffix.lower().lstrip(".")
        if suffix in {"jpg", "jpeg"}:
            mime = "image/jpeg"
        elif suffix == "png":
            mime = "image/png"
        elif suffix == "webp":
            mime = "image/webp"
        else:
            raise ValueError(f"Unsupported image format: {path.suffix}")
        data = path.read_bytes()
        encoded = base64.b64encode(data).decode("utf-8")
        return f"data:{mime};base64,{encoded}"

    def propose_edit(
        self,
        image_path: str,
        target_attribute: str,
        prior_plan: Optional[str] = None,
        critic_notes: Optional[str] = None,
    ) -> PlanResult:
        prior_text = f"Prior attempt: {prior_plan}" if prior_plan else "No prior attempts."
        critic_text = f"Critic notes: {critic_notes}" if critic_notes else "No critic notes."
        user_prompt = (
            f"Image path: {image_path}\n"
            f"Target percept: {target_attribute}\n"
            f"{prior_text}\n"
            f"{critic_text}\n"
            "Respond with JSON containing edit_plan and target_object."
        )
        image_url = self._image_to_data_url(image_path)

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.planner_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                },
            ],
            response_format={"type": "json_object"},
        )
        content = completion.choices[0].message.content
        data = json.loads(content) if content else {}
        return PlanResult(
            edit_plan=data.get("edit_plan", "No plan generated"),
            target_object=data.get("target_object", "object"),
        )

    def critique_generated(
        self,
        image_path: str,
        edited_image_path: str,
        edit_plan: str,
        target_object: str,
    ) -> GeneratedCritiqueResult:
        user_prompt = (
            f"Original image path: {image_path}\n"
            f"Edited image path: {edited_image_path}\n"
            f"Target object (do not change): {target_object}\n"
            f"Edit plan: {edit_plan}\n"
            "Respond with JSON containing is_realistic, is_minimal_edit and notes."
        )
        original_url = self._image_to_data_url(image_path)
        edited_url = self._image_to_data_url(edited_image_path)
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.critic_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {"type": "image_url", "image_url": {"url": original_url}},
                        {"type": "image_url", "image_url": {"url": edited_url}},
                    ],
                },
            ],
            response_format={"type": "json_object"},
        )
        content = completion.choices[0].message.content
        data = json.loads(content) if content else {}
        return GeneratedCritiqueResult(
            is_realistic=bool(data.get("is_realistic", False)),
            is_minimal_edit=bool(data.get("is_minimal_edit", False)),
            notes=data.get("notes", ""),
        )
