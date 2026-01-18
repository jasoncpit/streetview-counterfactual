import json
from dataclasses import dataclass
from typing import Optional

from openai import OpenAI


@dataclass
class PlanResult:
    edit_plan: str
    target_object: str


@dataclass
class CritiqueResult:
    is_realistic: bool
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

    def propose_edit(
        self,
        image_path: str,
        target_attribute: str,
        prior_plan: Optional[str] = None,
    ) -> PlanResult:
        prior_text = f"Prior attempt: {prior_plan}" if prior_plan else "No prior attempts."
        user_prompt = (
            f"Image path: {image_path}\n"
            f"Target percept: {target_attribute}\n"
            f"{prior_text}\n"
            "Respond with JSON containing edit_plan and target_object."
        )

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.planner_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
        )
        content = completion.choices[0].message.content
        data = json.loads(content) if content else {}
        return PlanResult(
            edit_plan=data.get("edit_plan", "No plan generated"),
            target_object=data.get("target_object", "object"),
        )

    def critique(self, edited_image_path: str, notes: str = "") -> CritiqueResult:
        user_prompt = (
            f"Edited image: {edited_image_path}\n"
            f"Context: {notes}\n"
            "Respond with JSON containing is_realistic and notes."
        )
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.critic_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
        )
        content = completion.choices[0].message.content
        data = json.loads(content) if content else {}
        return CritiqueResult(
            is_realistic=bool(data.get("is_realistic", False)),
            notes=data.get("notes", ""),
        )

