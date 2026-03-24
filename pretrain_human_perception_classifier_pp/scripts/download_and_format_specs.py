from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable

from huggingface_hub import snapshot_download


REPO_ID = "matiasqr/specs"
ATTR_MAP = {
    "safe": "safety",
    "lively": "lively",
    "wealthy": "wealthy",
    "beautiful": "beautiful",
    "boring": "boring",
    "depressing": "depressing",
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def workspace_root() -> Path:
    return Path(__file__).resolve().parents[1]


def data_dirs() -> tuple[Path, Path]:
    root = workspace_root()
    raw_dir = root / "data" / "raw" / "specs"
    formatted_dir = root / "data" / "formatted"
    raw_dir.parent.mkdir(parents=True, exist_ok=True)
    formatted_dir.mkdir(parents=True, exist_ok=True)
    return raw_dir, formatted_dir


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, object]], fieldnames: list[str]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def clean_value(value: str | None) -> str:
    if value is None:
        return ""
    return value.strip()


def download_dataset(raw_dir: Path) -> Path:
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=str(raw_dir),
    )
    return raw_dir


def load_metadata(raw_dir: Path) -> tuple[dict[str, dict[str, str]], dict[str, str]]:
    img_paths_rows = read_csv_rows(raw_dir / "svi" / "img_paths.csv")
    metadata_rows = read_csv_rows(raw_dir / "svi" / "metadata.csv")
    complexity_rows = read_csv_rows(raw_dir / "svi" / "visual_complexity_all.csv")

    by_uuid: dict[str, dict[str, str]] = {}
    image_number_to_uuid: dict[str, str] = {}

    for row in img_paths_rows:
        uuid = clean_value(row.get("uuid"))
        raw_path = clean_value(row.get("path"))
        if not uuid:
            continue
        rel_image_path = raw_path.removeprefix("data/")
        by_uuid[uuid] = {
            "uuid": uuid,
            "source_dataset_path": raw_path,
            "raw_snapshot_relpath": rel_image_path,
            "local_image_path": str(repo_root() / "pretrain_human_perception_classifier_pp" / "data" / "raw" / "specs" / rel_image_path),
        }

    for row in metadata_rows:
        uuid = clean_value(row.get("uuid"))
        if not uuid or uuid not in by_uuid:
            continue
        image_number = clean_value(row.get("Image number"))
        by_uuid[uuid].update(
            {
                "city": clean_value(row.get("city")),
                "relabeled_name": clean_value(row.get("Relabelled Name")),
                "image_number": image_number,
            }
        )
        if image_number:
            image_number_to_uuid[image_number] = uuid

    for row in complexity_rows:
        uuid = clean_value(row.get("uuid"))
        if not uuid or uuid not in by_uuid:
            continue
        by_uuid[uuid].update(
            {
                "visual_complexity": clean_value(row.get("visual_complexity")),
            }
        )

    return by_uuid, image_number_to_uuid


def load_inference_scores(raw_dir: Path) -> dict[str, dict[str, str]]:
    scores: dict[str, dict[str, str]] = defaultdict(dict)
    for source_name, target_name in ATTR_MAP.items():
        rows = read_csv_rows(raw_dir / "labels" / "inferences" / f"{source_name}.csv")
        for row in rows:
            uuid = clean_value(row.get("uuid"))
            if not uuid:
                continue
            scores[uuid][f"inference_{target_name}"] = clean_value(row.get(source_name))
    return scores


def load_qscores(
    raw_dir: Path,
    image_number_to_uuid: dict[str, str],
) -> dict[str, dict[str, str]]:
    qscores: dict[str, dict[str, str]] = defaultdict(dict)
    rows = read_csv_rows(raw_dir / "labels" / "processed" / "global_mapped_cleaned_qscores.csv")
    for row in rows:
        image_number = clean_value(row.get("Image"))
        source_question = clean_value(row.get("Question"))
        uuid = image_number_to_uuid.get(image_number)
        attr = ATTR_MAP.get(source_question)
        if not uuid or not attr:
            continue
        qscores[uuid][f"qscore_{attr}"] = clean_value(row.get("Score"))
        qscores[uuid][f"qscore_{attr}_num_comparisons"] = clean_value(row.get("Num_comparisons"))
    return qscores


def build_image_manifest(
    metadata_by_uuid: dict[str, dict[str, str]],
    inference_scores: dict[str, dict[str, str]],
    qscores: dict[str, dict[str, str]],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for uuid in sorted(metadata_by_uuid):
        row = {
            "uuid": uuid,
            "city": "",
            "image_number": "",
            "relabeled_name": "",
            "source_dataset_path": "",
            "raw_snapshot_relpath": "",
            "local_image_path": "",
            "visual_complexity": "",
        }
        row.update(metadata_by_uuid.get(uuid, {}))
        row.update(inference_scores.get(uuid, {}))
        row.update(qscores.get(uuid, {}))
        rows.append(row)
    return rows


def build_pairwise_rows(
    raw_dir: Path,
    image_number_to_uuid: dict[str, str],
    metadata_by_uuid: dict[str, dict[str, str]],
) -> list[dict[str, str]]:
    rows = read_csv_rows(raw_dir / "labels" / "processed" / "global_mapped_cleaned.csv")
    formatted_rows: list[dict[str, str]] = []
    for row in rows:
        left_image_number = clean_value(row.get("Left_image"))
        right_image_number = clean_value(row.get("Right_image"))
        left_uuid = image_number_to_uuid.get(left_image_number, "")
        right_uuid = image_number_to_uuid.get(right_image_number, "")
        question = ATTR_MAP.get(clean_value(row.get("Question")), clean_value(row.get("Question")))
        formatted_rows.append(
            {
                "respondent": clean_value(row.get("Respondent")),
                "country": clean_value(row.get("Country")),
                "question": question,
                "score": clean_value(row.get("Score")),
                "left_image_number": left_image_number,
                "left_uuid": left_uuid,
                "left_city": metadata_by_uuid.get(left_uuid, {}).get("city", ""),
                "left_local_image_path": metadata_by_uuid.get(left_uuid, {}).get("local_image_path", ""),
                "right_image_number": right_image_number,
                "right_uuid": right_uuid,
                "right_city": metadata_by_uuid.get(right_uuid, {}).get("city", ""),
                "right_local_image_path": metadata_by_uuid.get(right_uuid, {}).get("local_image_path", ""),
                "gender": clean_value(row.get("gender")),
                "age_group": clean_value(row.get("age_group")),
                "age_group_2": clean_value(row.get("age_group_2")),
                "nationality": clean_value(row.get("nationality")),
                "city_living": clean_value(row.get("city_living")),
                "city_living_length": clean_value(row.get("city_living_length")),
                "ahi": clean_value(row.get("ahi")),
                "ahi_2": clean_value(row.get("ahi_2")),
                "ahi_3": clean_value(row.get("ahi_3")),
                "num_household": clean_value(row.get("num_household")),
                "education_level": clean_value(row.get("education_level")),
                "education_level_remapped": clean_value(row.get("education_level_remapped")),
                "race_ethnicity": clean_value(row.get("race_ethnicity")),
                "extraversion": clean_value(row.get("extraversion")),
                "agreeableness": clean_value(row.get("agreeableness")),
                "conscientiousness": clean_value(row.get("conscientiousness")),
                "neuroticism": clean_value(row.get("neuroticism")),
                "openness": clean_value(row.get("openness")),
            }
        )
    return formatted_rows


def build_summary(
    raw_dir: Path,
    image_manifest: list[dict[str, str]],
    pairwise_rows: list[dict[str, str]],
) -> dict[str, object]:
    city_counts = Counter(row["city"] for row in image_manifest if row.get("city"))
    inference_coverage = {
        attr: sum(1 for row in image_manifest if row.get(f"inference_{attr}"))
        for attr in ATTR_MAP.values()
    }
    qscore_coverage = {
        attr: sum(1 for row in image_manifest if row.get(f"qscore_{attr}"))
        for attr in ATTR_MAP.values()
    }
    return {
        "dataset_repo_id": REPO_ID,
        "raw_snapshot_dir": str(raw_dir),
        "num_images": len(image_manifest),
        "num_pairwise_rows": len(pairwise_rows),
        "cities": dict(sorted(city_counts.items())),
        "inference_coverage": inference_coverage,
        "qscore_coverage": qscore_coverage,
    }


def main() -> None:
    raw_dir, formatted_dir = data_dirs()
    download_dataset(raw_dir)

    metadata_by_uuid, image_number_to_uuid = load_metadata(raw_dir)
    inference_scores = load_inference_scores(raw_dir)
    qscores = load_qscores(raw_dir, image_number_to_uuid)

    image_manifest = build_image_manifest(metadata_by_uuid, inference_scores, qscores)
    pairwise_rows = build_pairwise_rows(raw_dir, image_number_to_uuid, metadata_by_uuid)
    summary = build_summary(raw_dir, image_manifest, pairwise_rows)

    image_fieldnames = [
        "uuid",
        "city",
        "image_number",
        "relabeled_name",
        "source_dataset_path",
        "raw_snapshot_relpath",
        "local_image_path",
        "visual_complexity",
        "inference_safety",
        "inference_lively",
        "inference_wealthy",
        "inference_beautiful",
        "inference_boring",
        "inference_depressing",
        "qscore_safety",
        "qscore_safety_num_comparisons",
        "qscore_lively",
        "qscore_lively_num_comparisons",
        "qscore_wealthy",
        "qscore_wealthy_num_comparisons",
        "qscore_beautiful",
        "qscore_beautiful_num_comparisons",
        "qscore_boring",
        "qscore_boring_num_comparisons",
        "qscore_depressing",
        "qscore_depressing_num_comparisons",
    ]
    pairwise_fieldnames = [
        "respondent",
        "country",
        "question",
        "score",
        "left_image_number",
        "left_uuid",
        "left_city",
        "left_local_image_path",
        "right_image_number",
        "right_uuid",
        "right_city",
        "right_local_image_path",
        "gender",
        "age_group",
        "age_group_2",
        "nationality",
        "city_living",
        "city_living_length",
        "ahi",
        "ahi_2",
        "ahi_3",
        "num_household",
        "education_level",
        "education_level_remapped",
        "race_ethnicity",
        "extraversion",
        "agreeableness",
        "conscientiousness",
        "neuroticism",
        "openness",
    ]

    write_csv(formatted_dir / "specs_image_manifest.csv", image_manifest, image_fieldnames)
    write_csv(formatted_dir / "specs_pairwise_comparisons.csv", pairwise_rows, pairwise_fieldnames)
    (formatted_dir / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print(f"Downloaded raw snapshot to: {raw_dir}")
    print(f"Wrote image manifest: {formatted_dir / 'specs_image_manifest.csv'}")
    print(f"Wrote pairwise table: {formatted_dir / 'specs_pairwise_comparisons.csv'}")
    print(f"Wrote summary: {formatted_dir / 'dataset_summary.json'}")


if __name__ == "__main__":
    main()
