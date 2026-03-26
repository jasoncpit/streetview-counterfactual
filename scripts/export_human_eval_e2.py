from __future__ import annotations

import argparse
import csv
import hashlib
import random
import shutil
from pathlib import Path

from src.lever_identity import lever_identity_label


ATTRIBUTE_PROMPTS = {
    "safety": "Which image looks safer?",
    "safe": "Which image looks safer?",
    "lively": "Which image looks livelier?",
    "wealthy": "Which image looks wealthier?",
    "beautiful": "Which image looks more beautiful?",
    "boring": "Which image looks more boring?",
    "depressing": "Which image looks more depressing?",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export accepted candidate edits for E2 human pairwise evaluation.",
    )
    parser.add_argument("--csv", required=True, help="Accepted candidate CSV from generate_counterfactual.py")
    parser.add_argument(
        "--output-dir",
        default="data/06_human_eval_exports",
        help="Directory for the E2 export package.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed used for left/right assignment.",
    )
    return parser.parse_args()


def parse_bool(value: object) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(newline="", encoding="utf-8") as f:
        return [dict(row) for row in csv.DictReader(f)]


def question_text(attribute: str) -> str:
    key = str(attribute or "").strip().lower()
    return ATTRIBUTE_PROMPTS.get(key, f"Which image looks higher in {attribute}?")


def stable_pair_id(row: dict[str, str]) -> str:
    raw = "|".join(
        [
            str(row.get("input_image_path", "")),
            str(row.get("candidate_id", "")),
            str(row.get("target_attribute", "")),
            lever_identity_label(row),
        ]
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def copy_asset(source: Path, destination: Path) -> str:
    ensure_dir(destination.parent)
    shutil.copy2(source, destination)
    return destination.name


def build_manifest_rows(
    rows: list[dict[str, str]],
    *,
    csv_stem: str,
    assets_dir: Path,
    rng: random.Random,
) -> list[dict[str, str]]:
    manifest_rows: list[dict[str, str]] = []
    for row in rows:
        if not parse_bool(row.get("critic_is_valid")):
            continue

        input_path = Path(str(row.get("input_image_path", "")).strip())
        output_path = Path(str(row.get("output_image_path", "")).strip())
        if not input_path.exists() or not output_path.exists():
            continue

        pair_id = stable_pair_id(row)
        left_is_original = bool(rng.getrandbits(1))
        left_role = "original" if left_is_original else "edited"
        right_role = "edited" if left_is_original else "original"

        original_name = copy_asset(input_path, assets_dir / f"{pair_id}_original{input_path.suffix.lower()}")
        edited_name = copy_asset(output_path, assets_dir / f"{pair_id}_edited{output_path.suffix.lower()}")
        left_name = original_name if left_role == "original" else edited_name
        right_name = edited_name if right_role == "edited" else original_name

        manifest_rows.append(
            {
                "pair_id": pair_id,
                "source_csv_stem": csv_stem,
                "image_id": Path(str(row.get("input_image_path", ""))).stem,
                "candidate_id": str(row.get("candidate_id", "")),
                "target_attribute": str(row.get("target_attribute", "")),
                "question_text": question_text(str(row.get("target_attribute", ""))),
                "left_image_role": left_role,
                "right_image_role": right_role,
                "left_image_path": f"assets/{left_name}",
                "right_image_path": f"assets/{right_name}",
                "source_input_image_path": str(input_path),
                "source_output_image_path": str(output_path),
                "lever_identity_label": row.get("lever_identity_label", "") or lever_identity_label(row),
                "lever_concept": str(row.get("lever_concept", "")),
                "scene_support": str(row.get("scene_support", "")),
                "intervention_direction": str(row.get("intervention_direction", "")),
                "edit_template": str(row.get("edit_template", "")),
                "planner_target_object": str(row.get("planner_target_object", "")),
                "planner_edit_plan": str(row.get("planner_edit_plan", "")),
            }
        )
    return manifest_rows


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def response_template_rows(manifest_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        {
            "pair_id": row["pair_id"],
            "rater_id": "",
            "selected_side": "",
        }
        for row in manifest_rows
    ]


def main() -> None:
    args = parse_args()
    input_csv = Path(args.csv)
    rows = read_rows(input_csv)

    output_root = Path(args.output_dir) / input_csv.stem
    assets_dir = output_root / "assets"
    ensure_dir(assets_dir)

    manifest_rows = build_manifest_rows(
        rows,
        csv_stem=input_csv.stem,
        assets_dir=assets_dir,
        rng=random.Random(args.seed),
    )
    manifest_path = output_root / "e2_manifest.csv"
    responses_template_path = output_root / "e2_responses_template.csv"
    write_csv(
        manifest_path,
        manifest_rows,
        [
            "pair_id",
            "source_csv_stem",
            "image_id",
            "candidate_id",
            "target_attribute",
            "question_text",
            "left_image_role",
            "right_image_role",
            "left_image_path",
            "right_image_path",
            "source_input_image_path",
            "source_output_image_path",
            "lever_identity_label",
            "lever_concept",
            "scene_support",
            "intervention_direction",
            "edit_template",
            "planner_target_object",
            "planner_edit_plan",
        ],
    )
    write_csv(
        responses_template_path,
        response_template_rows(manifest_rows),
        ["pair_id", "rater_id", "selected_side"],
    )

    print(f"Wrote E2 manifest: {manifest_path}")
    print(f"Wrote response template: {responses_template_path}")
    print(f"Exported pairs: {len(manifest_rows)}")


if __name__ == "__main__":
    main()
