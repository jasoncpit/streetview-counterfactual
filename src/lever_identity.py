from __future__ import annotations

from typing import Any, Mapping


LEVER_IDENTITY_KEYS = (
    "lever_concept",
    "scene_support",
    "intervention_direction",
    "edit_template",
    "target_object",
)


def normalize_text(value: object) -> str:
    return " ".join(str(value or "").strip().split())


def normalize_ontology_value(value: object) -> str:
    return normalize_text(value).casefold()


def ontology_lookup(lever_ontology: tuple[str, ...]) -> dict[str, str]:
    return {
        normalize_ontology_value(item): normalize_text(item)
        for item in lever_ontology
        if normalize_text(item)
    }


def get_target_object(row: Mapping[str, Any]) -> str:
    return normalize_text(
        row.get("target_object")
        or row.get("planner_target_object")
        or row.get("planner_target_object_label")
        or ""
    )


def lever_identity_from_row(row: Mapping[str, Any]) -> dict[str, str]:
    return {
        "lever_concept": normalize_text(row.get("lever_concept")),
        "scene_support": normalize_text(row.get("scene_support")),
        "intervention_direction": normalize_text(row.get("intervention_direction")),
        "edit_template": normalize_text(row.get("edit_template")),
        "target_object": get_target_object(row),
    }


def lever_identity_label(row: Mapping[str, Any]) -> str:
    identity = lever_identity_from_row(row)
    pieces = [
        identity["lever_concept"],
        identity["intervention_direction"],
        identity["target_object"],
        identity["scene_support"],
    ]
    return " | ".join(piece for piece in pieces if piece)


def serialize_identity_values(
    rows: list[Mapping[str, Any]],
    key: str,
) -> str:
    seen: set[str] = set()
    values: list[str] = []
    for row in rows:
        value = lever_identity_from_row(row).get(key, "")
        if not value or value in seen:
            continue
        seen.add(value)
        values.append(value)
    return "|".join(values)


def serialize_identity_labels(rows: list[Mapping[str, Any]]) -> str:
    seen: set[str] = set()
    labels: list[str] = []
    for row in rows:
        label = lever_identity_label(row)
        if not label or label in seen:
            continue
        seen.add(label)
        labels.append(label)
    return "|".join(labels)
