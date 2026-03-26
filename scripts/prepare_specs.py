from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from pathlib import Path


SPECS_ROOT = Path("pretrain_human_perception_classifier_pp") / "data" / "raw" / "specs"
SVI_DIR = SPECS_ROOT / "svi"
INFERENCE_DIR = SPECS_ROOT / "labels" / "inferences"
METADATA_CSV = SVI_DIR / "metadata.csv"
IMG_PATHS_CSV = SVI_DIR / "img_paths.csv"
RAW_DIR = Path("data") / "01_raw"
PILOT_IDS_FILE = Path("pilot_ids.txt")
SCORES_CACHE = Path("data") / "specs_scores.json"

ATTRIBUTE_FILE_MAP = {
    "safe": "safe.csv",
    "lively": "lively.csv",
    "wealthy": "wealthy.csv",
    "beautiful": "beautiful.csv",
    "boring": "boring.csv",
    "depressing": "depressing.csv",
}

ID_COLS = ["image_id", "img_id", "id", "panoid", "pano_id", "uuid"]
SCORE_COLS = ["score", "safety_score", "prediction", "pred", "q_score", "inferred_score"]
PATH_COLS = ["image_path", "img_path", "path", "filename", "file"]
CITY_COLS = ["city", "country", "location"]


def find_col(headers: list[str], candidates: list[str]) -> str | None:
    header_lookup = {header.lower().strip(): header for header in headers}
    for candidate in candidates:
        match = header_lookup.get(candidate.lower())
        if match is not None:
            return match
    return None


def load_inference_scores(attribute: str = "safe") -> dict[str, float]:
    filename = ATTRIBUTE_FILE_MAP.get(attribute)
    if filename is None:
        raise ValueError(
            f"Unknown attribute '{attribute}'. Choose from: {sorted(ATTRIBUTE_FILE_MAP)}"
        )

    score_path = INFERENCE_DIR / filename
    if not score_path.exists():
        candidates = sorted(INFERENCE_DIR.glob(f"*{attribute}*.csv"))
        if not candidates:
            raise FileNotFoundError(
                f"No inference file found for attribute '{attribute}' in {INFERENCE_DIR}"
            )
        score_path = candidates[0]

    scores: dict[str, float] = {}
    with score_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        id_col = find_col(headers, ID_COLS)
        score_candidates = [attribute, *SCORE_COLS]
        score_col = find_col(headers, score_candidates)
        if id_col is None or score_col is None:
            raise ValueError(
                f"Cannot find ID or score columns in {score_path}. Headers: {headers}"
            )

        for row in reader:
            image_id = row.get(id_col, "").strip()
            if not image_id:
                continue
            try:
                scores[image_id] = float(row[score_col])
            except (TypeError, ValueError):
                continue

    print(f"  Loaded {len(scores)} scores from {score_path.name}")
    return scores


def load_metadata(city_filter: str | None = "singapore") -> list[dict[str, str]]:
    meta_path = METADATA_CSV if METADATA_CSV.exists() else IMG_PATHS_CSV
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Metadata file not found. Tried {METADATA_CSV} and {IMG_PATHS_CSV}."
        )

    rows: list[dict[str, str]] = []
    with meta_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        id_col = find_col(headers, ID_COLS)
        path_col = find_col(headers, PATH_COLS)
        city_col = find_col(headers, CITY_COLS)
        if id_col is None:
            raise ValueError(f"Cannot find image ID column in {meta_path}. Headers: {headers}")

        for row in reader:
            image_id = row.get(id_col, "").strip()
            if not image_id:
                continue

            image_path = row.get(path_col, "").strip() if path_col else ""
            if city_filter:
                city_value = row.get(city_col, "").lower() if city_col else ""
                path_value = image_path.lower()
                if city_filter.lower() not in city_value and city_filter.lower() not in path_value:
                    continue

            rows.append({"image_id": image_id, "image_path": image_path, **row})

    print(
        f"  Loaded {len(rows)} images from metadata"
        + (f" (city={city_filter})" if city_filter else "")
    )
    return rows


def stratified_sample(
    image_ids: list[str],
    scores: dict[str, float],
    n_per_stratum: int = 10,
    seed: int = 42,
) -> list[str]:
    scored = [(image_id, scores[image_id]) for image_id in image_ids if image_id in scores]
    if not scored:
        raise ValueError("No images found with matching scores.")

    scored.sort(key=lambda item: item[1])
    third = max(1, len(scored) // 3)
    strata = {
        "low": scored[:third],
        "mid": scored[third : 2 * third],
        "high": scored[2 * third :],
    }

    rng = random.Random(seed)
    sampled: list[str] = []
    for name, values in strata.items():
        k = min(n_per_stratum, len(values))
        chosen = rng.sample(values, k)
        sampled.extend(image_id for image_id, _ in chosen)
        preview = ", ".join(f"{score:.3f}" for _, score in chosen[:3])
        print(f"  Stratum '{name}': {len(values)} available, sampled {k} (e.g. {preview})")
    return sampled


def resolve_image_path(image_id: str, image_path_hint: str = "") -> Path | None:
    if image_path_hint:
        hint_path = Path(image_path_hint)
        if hint_path.exists():
            return hint_path
        specs_relative = SPECS_ROOT / image_path_hint.replace("data/", "", 1)
        if specs_relative.exists():
            return specs_relative
        svi_relative = SVI_DIR / Path(image_path_hint).name
        if svi_relative.exists():
            return svi_relative

    for ext in (".jpg", ".jpeg", ".png", ".webp"):
        direct_candidates = [
            SVI_DIR / f"{image_id}{ext}",
            SVI_DIR / "images" / f"{image_id}{ext}",
        ]
        for candidate in direct_candidates:
            if candidate.exists():
                return candidate
        matches = list(SVI_DIR.rglob(f"{image_id}{ext}"))
        if matches:
            return matches[0]
    return None


def materialize_images(
    sampled_ids: list[str],
    metadata: list[dict[str, str]],
    *,
    link_mode: str = "copy",
    dry_run: bool = False,
) -> dict[str, str]:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    metadata_by_id = {row["image_id"]: row for row in metadata}

    copied: dict[str, str] = {}
    missing: list[str] = []
    for image_id in sampled_ids:
        row = metadata_by_id.get(image_id, {})
        source = resolve_image_path(image_id, row.get("image_path", ""))
        if source is None:
            missing.append(image_id)
            continue

        destination = RAW_DIR / f"{image_id}{source.suffix.lower()}"
        if dry_run:
            copied[image_id] = str(destination)
            continue

        if destination.exists() or destination.is_symlink():
            destination.unlink()

        if link_mode == "symlink":
            destination.symlink_to(source.resolve())
        else:
            shutil.copy2(source, destination)
        copied[image_id] = str(destination)

    if missing:
        preview = ", ".join(missing[:10])
        print(f"  [warn] Missing {len(missing)} images: {preview}")
    print(f"  Materialized {len(copied)} images into {RAW_DIR}")
    return copied


def save_scores_cache(sampled_ids: list[str], scores: dict[str, float]) -> None:
    SCORES_CACHE.parent.mkdir(parents=True, exist_ok=True)
    payload = {image_id: scores.get(image_id) for image_id in sampled_ids}
    with SCORES_CACHE.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"  Saved baseline scores to {SCORES_CACHE}")


def write_pilot_ids(sampled_ids: list[str]) -> None:
    with PILOT_IDS_FILE.open("w", encoding="utf-8") as f:
        for image_id in sampled_ids:
            f.write(f"{image_id}\n")
    print(f"  Wrote {len(sampled_ids)} pilot IDs to {PILOT_IDS_FILE}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a SPECS pilot dataset for the pipeline.")
    parser.add_argument("--city", default="singapore", help="City filter for metadata.")
    parser.add_argument("--attribute", default="safe", help="Perception attribute for stratification.")
    parser.add_argument("--n-per-stratum", type=int, default=10, help="Images per low/mid/high stratum.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    parser.add_argument("--pilot-ids", type=str, default=None, help="Existing pilot_ids file to reuse.")
    parser.add_argument(
        "--link-mode",
        choices=("copy", "symlink"),
        default="copy",
        help="How to materialize pilot images into data/01_raw.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Plan only; do not write files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("\n-- SPECS Preparation --------------------------------------------------")
    print(f"\n[1/4] Loading ViT-PP2 scores for attribute '{args.attribute}'...")
    scores = load_inference_scores(args.attribute)

    print(f"\n[2/4] Loading metadata (city={args.city})...")
    metadata = load_metadata(args.city)
    all_ids = [row["image_id"] for row in metadata]

    if args.pilot_ids and Path(args.pilot_ids).exists():
        print(f"\n[3/4] Loading existing pilot IDs from {args.pilot_ids}...")
        with open(args.pilot_ids, encoding="utf-8") as f:
            sampled_ids = [line.strip() for line in f if line.strip()]
        print(f"  Loaded {len(sampled_ids)} IDs")
    else:
        print(f"\n[3/4] Stratified sampling ({args.n_per_stratum} per stratum)...")
        sampled_ids = stratified_sample(
            all_ids,
            scores,
            n_per_stratum=args.n_per_stratum,
            seed=args.seed,
        )
        if not args.dry_run:
            write_pilot_ids(sampled_ids)

    print(f"\n[4/4] Materializing {len(sampled_ids)} images into {RAW_DIR}...")
    if args.dry_run:
        print("  [dry-run] Skipping copy/symlink and score-cache writes")
    else:
        materialize_images(
            sampled_ids,
            metadata,
            link_mode=args.link_mode,
            dry_run=False,
        )
        save_scores_cache(sampled_ids, scores)

    print("\n-- Summary ------------------------------------------------------------")
    print(f"  Pilot size:    {len(sampled_ids)} images")
    print(f"  City:          {args.city}")
    print(f"  Attribute:     {args.attribute}")
    print(f"  pilot_ids.txt: {PILOT_IDS_FILE}")
    print(f"  Raw images:    {RAW_DIR}")
    print(f"  Score cache:   {SCORES_CACHE}")
    print(
        "\n  Next: uv run python scripts/generate_counterfactual.py "
        "--input-ids pilot_ids.txt\n"
    )


if __name__ == "__main__":
    main()
