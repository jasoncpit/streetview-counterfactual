import random
from pathlib import Path

from PIL import Image

from scripts.export_human_eval_e2 import build_manifest_rows
from scripts.ingest_human_eval_e2 import aggregate_pair_results, summarize_by_image


def make_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (12, 12), color="white").save(path)


def test_e2_export_and_ingest_round_trip(tmp_path):
    input_path = tmp_path / "input.jpg"
    output_path = tmp_path / "edited.jpg"
    make_image(input_path)
    make_image(output_path)

    manifest_rows = build_manifest_rows(
        [
            {
                "input_image_path": str(input_path),
                "output_image_path": str(output_path),
                "candidate_id": "1",
                "target_attribute": "safety",
                "lever_concept": "surface cleaning",
                "scene_support": "dirty sidewalk patches",
                "intervention_direction": "clean",
                "edit_template": "clean the sidewalk",
                "planner_target_object": "sidewalk",
                "planner_edit_plan": "Clean the sidewalk while preserving the rest of the scene.",
                "critic_is_valid": "true",
            }
        ],
        csv_stem="pilot",
        assets_dir=tmp_path / "assets",
        rng=random.Random(3),
    )

    assert len(manifest_rows) == 1
    assert manifest_rows[0]["question_text"] == "Which image looks safer?"

    pair_rows = aggregate_pair_results(
        manifest_rows,
        [{"pair_id": manifest_rows[0]["pair_id"], "selected_side": "edited"}],
        threshold=0.65,
    )
    image_rows = summarize_by_image(pair_rows)

    assert pair_rows[0]["edited_preference_share"] == 1.0
    assert pair_rows[0]["is_human_effective"] is True
    assert image_rows[0]["n_human_effective_levers"] == 1
    assert "surface cleaning | clean | sidewalk" in image_rows[0]["human_effective_lever_labels"]
