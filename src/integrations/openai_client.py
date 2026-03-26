from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass
from pathlib import Path

from openai import OpenAI

from src.lever_identity import ontology_lookup


@dataclass
class LeverCandidate:
    lever_concept: str
    scene_support: str
    target_object: str
    intervention_direction: str
    edit_template: str
    edit_plan: str


@dataclass
class CandidateSetResult:
    candidates: list[LeverCandidate]


@dataclass
class GeneratedCritiqueResult:
    same_place_preserved: bool
    is_localized: bool
    is_realistic: bool
    is_plausible: bool
    notes: str

    @property
    def is_valid(self) -> bool:
        return (
            self.same_place_preserved
            and self.is_localized
            and self.is_realistic
            and self.is_plausible
        )


class Planner:
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

    @staticmethod
    def _stringify_field(raw: object) -> str:
        if raw is None:
            return ""
        if isinstance(raw, list):
            return " ".join(str(item) for item in raw if item is not None)
        if isinstance(raw, dict):
            return " ".join(f"{key}: {value}" for key, value in raw.items())
        return str(raw)

    @classmethod
    def _sanitize_short_text(cls, raw: object, *, default: str, max_words: int = 8) -> str:
        if not raw:
            return default
        text = cls._stringify_field(raw).strip().replace("\n", " ")
        text = re.sub(r"^[\[\]\(\)\{\}\-\*\d\.\s:]+", "", text)
        text = re.sub(r"[\[\]\(\)\{\}]+", "", text)
        text = re.sub(r"\s+", " ", text)
        text = text.replace('"', "").replace("'", "").strip(" .,-")
        words = text.split()
        if not words:
            return default
        words = words[:max_words]
        trailing_stopwords = {
            "a",
            "an",
            "and",
            "at",
            "by",
            "for",
            "from",
            "in",
            "of",
            "on",
            "or",
            "the",
            "to",
            "with",
        }
        while len(words) > 1 and words[-1].lower() in trailing_stopwords:
            words.pop()
        text = " ".join(words).strip(" .,-")
        return text or default

    @classmethod
    def _sanitize_sentence(cls, raw: object, *, default: str, max_words: int = 28) -> str:
        text = cls._sanitize_short_text(raw, default=default, max_words=max_words)
        text = re.split(r"(?<=[.!?])\s+", text)[0].strip()
        text = text.rstrip(",;:")
        if not text:
            return default
        if text[-1] not in ".!?":
            text = f"{text}."
        return text

    @staticmethod
    def _is_valid_scene_support(value: str) -> bool:
        word_count = len(value.split())
        return 2 <= word_count <= 12

    def _coerce_candidate(
        self,
        data: dict,
        *,
        allowed_ontology: dict[str, str],
    ) -> LeverCandidate | None:
        raw_concept = self._sanitize_short_text(
            data.get("lever_concept"),
            default="",
            max_words=5,
        )
        lever_concept = allowed_ontology.get(raw_concept.casefold())
        if not lever_concept:
            return None
        scene_support = self._sanitize_short_text(
            data.get("scene_support") or data.get("target_object"),
            default="visible support",
            max_words=12,
        )
        if not self._is_valid_scene_support(scene_support):
            return None
        target_object = self._sanitize_short_text(
            data.get("target_object") or scene_support,
            default="object",
            max_words=5,
        )
        direction = self._sanitize_short_text(
            data.get("intervention_direction"),
            default="repair",
            max_words=3,
        )
        edit_template = self._sanitize_short_text(
            data.get("edit_template"),
            default=f"{direction} the {target_object}",
            max_words=12,
        )
        edit_plan = self._sanitize_sentence(
            data.get("edit_plan"),
            default=f"{direction.capitalize()} the {target_object}.",
            max_words=28,
        )
        return LeverCandidate(
            lever_concept=lever_concept,
            scene_support=scene_support,
            target_object=target_object,
            intervention_direction=direction,
            edit_template=edit_template,
            edit_plan=edit_plan,
        )

    def propose_lever_candidates(
        self,
        image_path: str,
        target_attribute: str,
        lever_ontology: tuple[str, ...],
        candidate_budget: int,
    ) -> CandidateSetResult:
        ontology_text = "\n".join(f"- {item}" for item in lever_ontology)
        allowed_ontology = ontology_lookup(lever_ontology)
        user_prompt = (
            f"Image path: {image_path}\n"
            f"Target percept: {target_attribute}\n"
            f"Candidate budget: {candidate_budget}\n"
            "Choose candidate levers only from this ontology:\n"
            f"{ontology_text}\n"
            "Return JSON with a top-level candidates array.\n"
            f"Target count: {candidate_budget} candidates.\n"
            "If possible, use different lever concepts across the returned candidates."
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
        raw_candidates = data.get("candidates")
        candidates = []
        seen_keys: set[tuple[str, str, str]] = set()
        if isinstance(raw_candidates, list):
            for item in raw_candidates:
                if isinstance(item, dict):
                    candidate = self._coerce_candidate(
                        item,
                        allowed_ontology=allowed_ontology,
                    )
                    if candidate is None:
                        continue
                    dedupe_key = (
                        candidate.lever_concept.lower(),
                        candidate.target_object.lower(),
                        candidate.edit_plan.lower(),
                    )
                    if dedupe_key in seen_keys:
                        continue
                    seen_keys.add(dedupe_key)
                    candidates.append(candidate)
                    if len(candidates) >= candidate_budget:
                        break
        return CandidateSetResult(candidates=candidates)

    def critique_generated(
        self,
        image_path: str,
        edited_image_path: str,
        edit_plan: str,
        target_object: str,
        lever_concept: str = "",
        scene_support: str = "",
        intervention_direction: str = "",
        edit_template: str = "",
    ) -> GeneratedCritiqueResult:
        user_prompt = (
            f"Original image path: {image_path}\n"
            f"Edited image path: {edited_image_path}\n"
            f"Lever concept: {lever_concept}\n"
            f"Scene support: {scene_support}\n"
            f"Target object (do not change): {target_object}\n"
            f"Intervention direction: {intervention_direction}\n"
            f"Edit template: {edit_template}\n"
            f"Edit plan: {edit_plan}\n"
            "Respond with JSON containing same_place_preserved, is_localized, is_realistic, is_plausible and notes."
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
        same_place_preserved = bool(
            data.get("same_place_preserved", data.get("is_minimal_edit", False))
        )
        is_localized = bool(
            data.get("is_localized", data.get("is_minimal_edit", False))
        )
        return GeneratedCritiqueResult(
            same_place_preserved=same_place_preserved,
            is_localized=is_localized,
            is_realistic=bool(data.get("is_realistic", False)),
            is_plausible=bool(data.get("is_plausible", data.get("is_realistic", False))),
            notes=data.get("notes", ""),
        )
