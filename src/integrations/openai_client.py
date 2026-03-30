from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from openai import OpenAI

from src.lever_identity import lever_family_lookup, normalize_family_value, ontology_lookup


@dataclass
class LeverCandidate:
    lever_concept: str
    lever_family: str
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
    edit_attempted: bool
    same_place_preserved: bool
    is_localized: bool
    is_realistic: bool
    is_plausible: bool
    failure_modes: list[str]
    diagnosis: str
    repair_suggestion: str

    @property
    def is_valid(self) -> bool:
        return (
            self.edit_attempted
            and self.same_place_preserved
            and self.is_localized
            and self.is_realistic
            and self.is_plausible
        )

    @property
    def notes(self) -> str:
        return f"{self.diagnosis} | repair: {self.repair_suggestion}"


class Planner:
    """
    Lightweight wrapper over OpenAI Chat Completions for planning and realism checks.
    """

    def __init__(
        self,
        model: str,
        planner_prompt: str,
        critic_prompt: str,
        planner_debug_hook: Callable[[dict[str, Any]], None] | None = None,
        critique_debug_hook: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.client = OpenAI()
        self.model = model
        self.planner_prompt = planner_prompt.strip()
        self.critic_prompt = critic_prompt.strip()
        self.planner_debug_hook = planner_debug_hook
        self.critique_debug_hook = critique_debug_hook

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
    def _normalize_failure_modes(raw: object) -> list[str]:
        if not isinstance(raw, list):
            return []
        values: list[str] = []
        for item in raw:
            text = " ".join(str(item or "").strip().split())
            if text:
                values.append(text)
        return values

    @classmethod
    def _normalize_critic_text(cls, raw: object, *, default: str) -> str:
        text = cls._sanitize_sentence(raw, default=default)
        normalized = text.strip().rstrip(".").casefold()
        if normalized in {"pass", "none"}:
            return normalized
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
        rejection_hook: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> LeverCandidate | None:
        family_lookup = lever_family_lookup()
        raw_concept = self._sanitize_short_text(
            data.get("lever_concept"),
            default="",
            max_words=5,
        )
        lever_concept = allowed_ontology.get(raw_concept.casefold())
        if not lever_concept:
            if rejection_hook is not None:
                rejection_hook("invalid lever_concept", data)
            return None
        lever_family = family_lookup.get(lever_concept.casefold(), "")
        raw_family = normalize_family_value(data.get("lever_family"))
        if not lever_family or raw_family != normalize_family_value(lever_family):
            if rejection_hook is not None:
                rejection_hook(
                    f"lever_family mismatch: expected '{lever_family}' got '{data.get('lever_family', '')}'",
                    data,
                )
            return None
        scene_support = self._sanitize_short_text(
            data.get("scene_support") or data.get("target_object"),
            default="visible support",
            max_words=12,
        )
        if not self._is_valid_scene_support(scene_support):
            if rejection_hook is not None:
                rejection_hook("invalid scene_support", data)
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
            lever_family=lever_family,
            scene_support=scene_support,
            target_object=target_object,
            intervention_direction=direction,
            edit_template=edit_template,
            edit_plan=edit_plan,
        )

    def _parse_candidates_from_content(
        self,
        content: str | None,
        *,
        allowed_ontology: dict[str, str],
        candidate_budget: int,
        debug_context: dict[str, Any] | None = None,
    ) -> list[LeverCandidate]:
        data = json.loads(content) if content else {}
        raw_candidates = data.get("candidates")
        candidates = []
        seen_keys: set[tuple[str, str, str]] = set()
        rejection_events: list[dict[str, Any]] = []
        if isinstance(raw_candidates, list):
            for item in raw_candidates:
                if isinstance(item, dict):
                    def rejection_hook(reason: str, raw_candidate: dict[str, Any]) -> None:
                        rejection_events.append(
                            {
                                "reason": reason,
                                "raw_candidate": raw_candidate,
                            }
                        )

                    candidate = self._coerce_candidate(
                        item,
                        allowed_ontology=allowed_ontology,
                        rejection_hook=rejection_hook,
                    )
                    if candidate is None:
                        continue
                    dedupe_key = (
                        candidate.lever_concept.lower(),
                        candidate.target_object.lower(),
                        candidate.edit_plan.lower(),
                    )
                    if dedupe_key in seen_keys:
                        rejection_events.append(
                            {
                                "reason": "duplicate candidate",
                                "raw_candidate": item,
                            }
                        )
                        continue
                    seen_keys.add(dedupe_key)
                    candidates.append(candidate)
                    if len(candidates) >= candidate_budget:
                        break
        if self.planner_debug_hook is not None:
            payload = {
                "event": "planner_parse_summary",
                "response_content": content or "",
                "accepted_candidates": [
                    {
                        "lever_concept": candidate.lever_concept,
                        "lever_family": candidate.lever_family,
                        "scene_support": candidate.scene_support,
                        "target_object": candidate.target_object,
                        "intervention_direction": candidate.intervention_direction,
                        "edit_template": candidate.edit_template,
                        "edit_plan": candidate.edit_plan,
                    }
                    for candidate in candidates
                ],
                "rejection_events": rejection_events,
            }
            if debug_context:
                payload.update(debug_context)
            self.planner_debug_hook(payload)
        return candidates

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
        if self.planner_debug_hook is not None:
            self.planner_debug_hook(
                {
                    "event": "planner_propose_exchange",
                    "system_prompt": self.planner_prompt,
                    "user_prompt": user_prompt,
                    "response_content": content or "",
                    "image_path": image_path,
                    "target_attribute": target_attribute,
                    "candidate_budget": candidate_budget,
                }
            )
        candidates = self._parse_candidates_from_content(
            content,
            allowed_ontology=allowed_ontology,
            candidate_budget=candidate_budget,
            debug_context={
                "event": "planner_propose_parse",
                "image_path": image_path,
                "target_attribute": target_attribute,
                "candidate_budget": candidate_budget,
            },
        )
        return CandidateSetResult(candidates=candidates)

    def revise_candidate_from_critique(
        self,
        image_path: str,
        target_attribute: str,
        candidate: LeverCandidate,
        diagnosis: str,
        repair_suggestion: str,
    ) -> LeverCandidate:
        allowed_ontology = ontology_lookup((candidate.lever_concept,))
        user_prompt = (
            f"Image path: {image_path}\n"
            f"Target percept: {target_attribute}\n"
            "Revise the CURRENT candidate using the critic feedback below.\n"
            "Keep the same lever concept and stay localized to the same support.\n"
            "Do not switch to a different lever or propose a global scene change.\n"
            f"Locked lever concept: {candidate.lever_concept}\n"
            f"Locked lever family: {candidate.lever_family}\n"
            f"Current scene support: {candidate.scene_support}\n"
            f"Current target object: {candidate.target_object}\n"
            f"Current intervention direction: {candidate.intervention_direction}\n"
            f"Current edit template: {candidate.edit_template}\n"
            f"Current edit plan: {candidate.edit_plan}\n"
            f"Critic diagnosis: {diagnosis}\n"
            f"Critic repair suggestion: {repair_suggestion}\n"
            "Return JSON with a top-level candidates array containing exactly one revised candidate.\n"
            "The candidate must contain exactly these fields:\n"
            "- lever_concept\n"
            "- lever_family\n"
            "- scene_support\n"
            "- target_object\n"
            "- intervention_direction\n"
            "- edit_template\n"
            "- edit_plan\n"
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
        if self.planner_debug_hook is not None:
            self.planner_debug_hook(
                {
                    "event": "planner_revise_exchange",
                    "system_prompt": self.planner_prompt,
                    "user_prompt": user_prompt,
                    "response_content": content or "",
                    "image_path": image_path,
                    "target_attribute": target_attribute,
                    "locked_lever_concept": candidate.lever_concept,
                    "locked_lever_family": candidate.lever_family,
                }
            )
        candidates = self._parse_candidates_from_content(
            content,
            allowed_ontology=allowed_ontology,
            candidate_budget=1,
            debug_context={
                "event": "planner_revise_parse",
                "image_path": image_path,
                "target_attribute": target_attribute,
                "locked_lever_concept": candidate.lever_concept,
                "locked_lever_family": candidate.lever_family,
            },
        )
        if not candidates:
            return candidate
        revised = candidates[0]
        if revised.lever_concept != candidate.lever_concept:
            return candidate
        return revised

    def critique_generated(
        self,
        image_path: str,
        edited_image_path: str,
        edit_plan: str,
        target_object: str,
        lever_concept: str = "",
        lever_family: str = "",
        scene_support: str = "",
        intervention_direction: str = "",
        edit_template: str = "",
        debug_context: dict[str, Any] | None = None,
    ) -> GeneratedCritiqueResult:
        user_prompt = (
            f"Original image path: {image_path}\n"
            f"Edited image path: {edited_image_path}\n"
            f"Lever concept: {lever_concept}\n"
            f"Lever family: {lever_family}\n"
            f"Scene support: {scene_support}\n"
            f"Target object (do not change): {target_object}\n"
            f"Intervention direction: {intervention_direction}\n"
            f"Edit template: {edit_template}\n"
            f"Edit plan: {edit_plan}\n"
            "Respond with JSON containing edit_attempted, same_place_preserved, is_localized, is_realistic, is_plausible and notes."
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
        if self.critique_debug_hook is not None:
            payload = {
                "system_prompt": self.critic_prompt,
                "user_prompt": user_prompt,
                "response_content": content or "",
            }
            if debug_context:
                payload.update(debug_context)
            self.critique_debug_hook(payload)
        data = json.loads(content) if content else {}
        notes_raw = data.get("notes", {})
        if isinstance(notes_raw, str):
            failure_modes = []
            diagnosis = self._normalize_critic_text(notes_raw, default="No diagnosis provided.")
            repair_suggestion = "none"
        else:
            failure_modes = self._normalize_failure_modes(
                notes_raw.get("failure_modes") if isinstance(notes_raw, dict) else []
            )
            diagnosis = self._normalize_critic_text(
                notes_raw.get("diagnosis") if isinstance(notes_raw, dict) else "",
                default="No diagnosis provided.",
            )
            repair_suggestion = self._normalize_critic_text(
                notes_raw.get("repair_suggestion") if isinstance(notes_raw, dict) else "",
                default="none",
            )
        same_place_preserved = bool(
            data.get("same_place_preserved", data.get("is_minimal_edit", False))
        )
        is_localized = bool(
            data.get("is_localized", data.get("is_minimal_edit", False))
        )
        return GeneratedCritiqueResult(
            edit_attempted=bool(data.get("edit_attempted", True)),
            same_place_preserved=same_place_preserved,
            is_localized=is_localized,
            is_realistic=bool(data.get("is_realistic", False)),
            is_plausible=bool(data.get("is_plausible", data.get("is_realistic", False))),
            failure_modes=failure_modes,
            diagnosis=diagnosis,
            repair_suggestion=repair_suggestion,
        )
