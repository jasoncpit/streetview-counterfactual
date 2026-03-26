from src import scoring


def test_score_candidate_rows_and_aux_summary_use_explicit_auxiliary_fields(monkeypatch):
    rows = [
        {
            "input_image_path": "data/01_raw/example.jpg",
            "output_image_path": "data/02_counterfactual/example_edit.jpg",
            "lever_concept": "surface cleaning",
            "scene_support": "dirty sidewalk patches",
            "intervention_direction": "clean",
            "edit_template": "clean the sidewalk",
            "planner_target_object": "sidewalk",
            "critic_is_valid": True,
        }
    ]

    def fake_score(image_path: str, attribute: str = "safety", *, device: str = "auto"):
        if image_path.endswith("example_edit.jpg"):
            return 6.0
        return 5.0

    monkeypatch.setattr(scoring, "_score_image_with_vitpp2", fake_score)

    scored_rows = scoring.score_candidate_rows(
        rows,
        attribute="safety",
        baseline_scores={"example": 5.0},
        theta=0.5,
    )
    summary = scoring.compute_per_image_aux_summary(scored_rows, theta=0.5)

    assert scored_rows[0]["auxiliary_threshold"] == 0.5
    assert scored_rows[0]["exceeds_auxiliary_threshold"] is True
    assert scored_rows[0]["lever_identity_label"] == (
        "surface cleaning | clean | sidewalk | dirty sidewalk patches"
    )

    assert summary["example"]["n_auxiliary_effective_levers"] == 1
    assert summary["example"]["auxiliary_effective_lever_concepts"] == "surface cleaning"
    assert summary["example"]["auxiliary_effective_target_objects"] == "sidewalk"
    assert "surface cleaning | clean | sidewalk" in summary["example"]["auxiliary_effective_lever_labels"]
