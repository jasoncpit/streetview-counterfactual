from src.integrations.openai_client import Planner
from src.lever_identity import ontology_lookup


def build_planner(monkeypatch) -> Planner:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    return Planner(model="gpt-5.2", planner_prompt="planner", critic_prompt="critic")


def test_coerce_candidate_accepts_exact_ontology_member(monkeypatch):
    planner = build_planner(monkeypatch)
    candidate = planner._coerce_candidate(
        {
            "lever_concept": "Surface Cleaning",
            "scene_support": "dirty sidewalk patches",
            "target_object": "sidewalk",
            "intervention_direction": "clean",
            "edit_template": "clean the sidewalk",
            "edit_plan": "Clean the sidewalk while preserving the rest of the scene.",
        },
        allowed_ontology=ontology_lookup(("surface cleaning", "graffiti removal")),
    )

    assert candidate is not None
    assert candidate.lever_concept == "surface cleaning"


def test_coerce_candidate_rejects_nonmember_ontology_value(monkeypatch):
    planner = build_planner(monkeypatch)
    candidate = planner._coerce_candidate(
        {
            "lever_concept": "trash cleanup",
            "scene_support": "litter around curb edge",
            "target_object": "curb",
            "intervention_direction": "clean",
            "edit_template": "clean the curb",
            "edit_plan": "Clean the curb while preserving the rest of the scene.",
        },
        allowed_ontology=ontology_lookup(("surface cleaning", "graffiti removal")),
    )

    assert candidate is None
