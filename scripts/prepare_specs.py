from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path


SPECS_ROOT = Path("pretrain_human_perception_classifier_pp") / "data" / "raw" / "specs"
SVI_DIR = SPECS_ROOT / "svi"
INFERENCE_DIR = SPECS_ROOT / "labels" / "inferences"
METADATA_CSV = SVI_DIR / "metadata.csv"
IMG_PATHS_CSV = SVI_DIR / "img_paths.csv"
VISUAL_COMPLEXITY_CSV = SVI_DIR / "visual_complexity_all.csv"
DEFAULT_RAW_DIR = Path("data") / "01_raw" / "specs_paper_n100"
DEFAULT_IDS_FILE = Path("paper_ids.txt")
DEFAULT_MANIFEST_PATH = Path("data") / "specs_paper_n100_manifest.csv"
DEFAULT_SUMMARY_PATH = Path("data") / "specs_paper_n100_summary.json"
DEFAULT_SCORES_CACHE = Path("data") / "specs_scores.json"

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


def normalize_city_name(value: str) -> str:
    return " ".join(value.strip().split())


def parse_city_arg(raw: str) -> set[str] | None:
    value = (raw or "").strip()
    if not value or value.lower() == "all":
        return None
    return {normalize_city_name(part).casefold() for part in value.split(",") if part.strip()}


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


def load_visual_complexity() -> dict[str, float]:
    if not VISUAL_COMPLEXITY_CSV.exists():
        raise FileNotFoundError(f"Visual complexity file not found: {VISUAL_COMPLEXITY_CSV}")

    values: dict[str, float] = {}
    with VISUAL_COMPLEXITY_CSV.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        id_col = find_col(headers, ID_COLS)
        score_col = find_col(headers, ["visual_complexity"])
        if id_col is None or score_col is None:
            raise ValueError(
                f"Cannot find UUID or complexity columns in {VISUAL_COMPLEXITY_CSV}. Headers: {headers}"
            )
        for row in reader:
            image_id = row.get(id_col, "").strip()
            if not image_id:
                continue
            try:
                values[image_id] = float(row[score_col])
            except (TypeError, ValueError):
                continue

    print(f"  Loaded {len(values)} visual complexity values")
    return values


def load_metadata(city_filter: set[str] | None = None) -> list[dict[str, str]]:
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
            city_value = normalize_city_name(row.get(city_col, "")) if city_col else ""
            if city_filter and city_value.casefold() not in city_filter:
                continue
            image_path = row.get(path_col, "").strip() if path_col else ""
            rows.append(
                {
                    "image_id": image_id,
                    "image_path": image_path,
                    "city": city_value,
                    **row,
                }
            )

    city_text = "all cities" if city_filter is None else ", ".join(sorted(city_filter))
    print(f"  Loaded {len(rows)} metadata rows ({city_text})")
    return rows


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


def assign_rank_bins(rows: list[dict[str, object]], value_key: str, bin_key: str, n_bins: int) -> None:
    if not rows:
        return
    ordered = sorted(rows, key=lambda row: float(row[value_key]))
    total = len(ordered)
    for idx, row in enumerate(ordered):
        row[bin_key] = min(n_bins - 1, int(idx * n_bins / total))


def allocate_per_city(city_names: list[str], n_total: int) -> dict[str, int]:
    if not city_names:
        raise ValueError("No cities available for allocation.")
    base = n_total // len(city_names)
    remainder = n_total % len(city_names)
    allocation = {city: base for city in city_names}
    for city in city_names[:remainder]:
        allocation[city] += 1
    return allocation


def sample_city_rows(
    city_rows: list[dict[str, object]],
    *,
    target_n: int,
    score_bins: int,
    complexity_bins: int,
    seed: int,
) -> list[dict[str, object]]:
    if target_n > len(city_rows):
        raise ValueError(
            f"Requested {target_n} rows from city pool of size {len(city_rows)}."
        )

    working_rows = [dict(row) for row in city_rows]
    assign_rank_bins(working_rows, "score", "score_bin", score_bins)
    assign_rank_bins(working_rows, "visual_complexity", "complexity_bin", complexity_bins)

    grouped: dict[tuple[int, int], list[dict[str, object]]] = defaultdict(list)
    for row in working_rows:
        grouped[(int(row["score_bin"]), int(row["complexity_bin"]))].append(row)

    rng = random.Random(seed)
    selected: list[dict[str, object]] = []
    cells = sorted(grouped)
    base = target_n // len(cells)
    remainder = target_n % len(cells)

    leftovers: list[dict[str, object]] = []
    for cell_idx, cell in enumerate(cells):
        bucket = grouped[cell]
        rng.shuffle(bucket)
        take = min(len(bucket), base + (1 if cell_idx < remainder else 0))
        selected.extend(bucket[:take])
        leftovers.extend(bucket[take:])

    if len(selected) < target_n:
        rng.shuffle(leftovers)
        needed = target_n - len(selected)
        selected.extend(leftovers[:needed])

    if len(selected) != target_n:
        raise ValueError(
            f"Could not sample exactly {target_n} rows for city {city_rows[0]['city']}."
        )
    return selected


def build_candidate_pool(
    metadata: list[dict[str, str]],
    *,
    scores: dict[str, float],
    visual_complexity: dict[str, float],
) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    missing_paths = 0
    for row in metadata:
        image_id = row["image_id"]
        score = scores.get(image_id)
        complexity = visual_complexity.get(image_id)
        if score is None or complexity is None:
            continue
        source_path = resolve_image_path(image_id, row.get("image_path", ""))
        if source_path is None:
            missing_paths += 1
            continue
        candidates.append(
            {
                "image_id": image_id,
                "city": row["city"],
                "image_path_hint": row.get("image_path", ""),
                "source_path": str(source_path),
                "score": float(score),
                "visual_complexity": float(complexity),
            }
        )

    if missing_paths:
        print(f"  [warn] Skipped {missing_paths} rows with no resolvable local image path")
    if not candidates:
        raise ValueError("No candidates with score, complexity, and local image path.")
    return candidates


def select_paper_subset(
    metadata: list[dict[str, str]],
    *,
    scores: dict[str, float],
    visual_complexity: dict[str, float],
    n_total: int,
    score_bins: int,
    complexity_bins: int,
    seed: int,
) -> list[dict[str, object]]:
    candidates = build_candidate_pool(
        metadata,
        scores=scores,
        visual_complexity=visual_complexity,
    )

    by_city: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in candidates:
        by_city[str(row["city"])].append(row)

    city_names = sorted(by_city)
    allocation = allocate_per_city(city_names, n_total)
    selected: list[dict[str, object]] = []
    for city_idx, city in enumerate(city_names):
        city_rows = by_city[city]
        target_n = allocation[city]
        chosen = sample_city_rows(
            city_rows,
            target_n=target_n,
            score_bins=score_bins,
            complexity_bins=complexity_bins,
            seed=seed + city_idx,
        )
        selected.extend(chosen)
        print(
            f"  City '{city}': {len(city_rows)} eligible, selected {len(chosen)} "
            f"using {score_bins}x{complexity_bins} vision-based strata"
        )

    if len(selected) != n_total:
        raise ValueError(f"Expected {n_total} selected rows, got {len(selected)}.")
    return sorted(selected, key=lambda row: (str(row["city"]), str(row["image_id"])))


def clean_output_dir(output_dir: Path) -> None:
    if not output_dir.exists():
        return
    for child in output_dir.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def materialize_images(
    selected_rows: list[dict[str, object]],
    *,
    output_dir: Path,
    link_mode: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for row in selected_rows:
        source = Path(str(row["source_path"]))
        destination = output_dir / f"{row['image_id']}{source.suffix.lower()}"
        if destination.exists() or destination.is_symlink():
            destination.unlink()
        if link_mode == "symlink":
            destination.symlink_to(source.resolve())
        else:
            shutil.copy2(source, destination)
        row["materialized_path"] = str(destination)


def write_ids_file(ids_file: Path, selected_rows: list[dict[str, object]]) -> None:
    ids_file.parent.mkdir(parents=True, exist_ok=True)
    with ids_file.open("w", encoding="utf-8") as f:
        for row in selected_rows:
            f.write(f"{row['image_id']}\n")


def write_manifest(manifest_path: Path, selected_rows: list[dict[str, object]]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image_id",
        "city",
        "score",
        "score_bin",
        "visual_complexity",
        "complexity_bin",
        "source_path",
        "materialized_path",
    ]
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in selected_rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_summary(summary_path: Path, selected_rows: list[dict[str, object]], *, attribute: str) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    by_city: dict[str, int] = defaultdict(int)
    for row in selected_rows:
        by_city[str(row["city"])] += 1
    payload = {
        "attribute": attribute,
        "n_selected": len(selected_rows),
        "cities": dict(sorted(by_city.items())),
        "score_min": min(float(row["score"]) for row in selected_rows),
        "score_max": max(float(row["score"]) for row in selected_rows),
        "visual_complexity_min": min(float(row["visual_complexity"]) for row in selected_rows),
        "visual_complexity_max": max(float(row["visual_complexity"]) for row in selected_rows),
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def save_scores_cache(selected_rows: list[dict[str, object]], cache_path: Path) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {str(row["image_id"]): float(row["score"]) for row in selected_rows}
    with cache_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_existing_ids(ids_path: Path, metadata: list[dict[str, str]]) -> list[str]:
    with ids_path.open(encoding="utf-8") as f:
        sampled_ids = [line.strip() for line in f if line.strip()]
    known_ids = {row["image_id"] for row in metadata}
    missing = [image_id for image_id in sampled_ids if image_id not in known_ids]
    if missing:
        preview = ", ".join(missing[:10])
        raise ValueError(f"IDs not found in filtered metadata: {preview}")
    return sampled_ids


def attach_existing_selection(
    sampled_ids: list[str],
    metadata: list[dict[str, str]],
    *,
    scores: dict[str, float],
    visual_complexity: dict[str, float],
) -> list[dict[str, object]]:
    metadata_by_id = {row["image_id"]: row for row in metadata}
    selected_rows: list[dict[str, object]] = []
    for image_id in sampled_ids:
        row = metadata_by_id[image_id]
        source_path = resolve_image_path(image_id, row.get("image_path", ""))
        if source_path is None:
            raise FileNotFoundError(f"Could not resolve image path for {image_id}")
        if image_id not in scores or image_id not in visual_complexity:
            raise ValueError(f"Missing score or visual complexity for {image_id}")
        selected_rows.append(
            {
                "image_id": image_id,
                "city": row["city"],
                "source_path": str(source_path),
                "score": float(scores[image_id]),
                "visual_complexity": float(visual_complexity[image_id]),
                "score_bin": "",
                "complexity_bin": "",
            }
        )
    return selected_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a paper-facing SPECS subset using vision-based stratification."
    )
    parser.add_argument(
        "--city",
        default="all",
        help="City filter: 'all', a single city, or a comma-separated city list.",
    )
    parser.add_argument(
        "--attribute",
        default="safe",
        help="Perception attribute whose inferred score drives the main stratification.",
    )
    parser.add_argument("--n-total", type=int, default=100, help="Total number of images to select.")
    parser.add_argument(
        "--score-bins",
        type=int,
        default=5,
        help="Number of per-city bins for the vision-model perceptual score.",
    )
    parser.add_argument(
        "--complexity-bins",
        type=int,
        default=2,
        help="Number of per-city bins for visual complexity.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for stratified sampling.")
    parser.add_argument(
        "--pilot-ids",
        type=str,
        default=None,
        help="Optional existing IDs file to reuse instead of drawing a new sample.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_RAW_DIR),
        help="Directory where the selected images should be materialized.",
    )
    parser.add_argument(
        "--ids-file",
        type=str,
        default=str(DEFAULT_IDS_FILE),
        help="Path for the selected image ID list.",
    )
    parser.add_argument(
        "--manifest-path",
        type=str,
        default=str(DEFAULT_MANIFEST_PATH),
        help="CSV manifest describing the selected subset.",
    )
    parser.add_argument(
        "--summary-path",
        type=str,
        default=str(DEFAULT_SUMMARY_PATH),
        help="JSON summary describing the selected subset.",
    )
    parser.add_argument(
        "--score-cache-path",
        type=str,
        default=str(DEFAULT_SCORES_CACHE),
        help="Baseline score cache written for downstream auxiliary scoring.",
    )
    parser.add_argument(
        "--link-mode",
        choices=("copy", "symlink"),
        default="copy",
        help="How to materialize selected images into the output directory.",
    )
    parser.add_argument(
        "--clean-output-dir",
        action="store_true",
        help="Remove existing contents of the output directory before materializing the new subset.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Plan only; do not write files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    city_filter = parse_city_arg(args.city)
    output_dir = Path(args.output_dir)
    ids_file = Path(args.ids_file)
    manifest_path = Path(args.manifest_path)
    summary_path = Path(args.summary_path)
    score_cache_path = Path(args.score_cache_path)

    print("\n-- SPECS Paper Subset Preparation ------------------------------------")
    print(f"\n[1/5] Loading inferred '{args.attribute}' scores...")
    scores = load_inference_scores(args.attribute)

    print("\n[2/5] Loading visual complexity scores...")
    visual_complexity = load_visual_complexity()

    print(f"\n[3/5] Loading metadata (city={args.city})...")
    metadata = load_metadata(city_filter)

    if args.pilot_ids and Path(args.pilot_ids).exists():
        print(f"\n[4/5] Reusing IDs from {args.pilot_ids}...")
        sampled_ids = load_existing_ids(Path(args.pilot_ids), metadata)
        selected_rows = attach_existing_selection(
            sampled_ids,
            metadata,
            scores=scores,
            visual_complexity=visual_complexity,
        )
    else:
        print(
            f"\n[4/5] Selecting N={args.n_total} using per-city score bins={args.score_bins} "
            f"and complexity bins={args.complexity_bins}..."
        )
        selected_rows = select_paper_subset(
            metadata,
            scores=scores,
            visual_complexity=visual_complexity,
            n_total=args.n_total,
            score_bins=args.score_bins,
            complexity_bins=args.complexity_bins,
            seed=args.seed,
        )

    print(f"\n[5/5] Preparing outputs in {output_dir}...")
    if args.dry_run:
        print("  [dry-run] Skipping cleanup, materialization, IDs, manifest, and cache writes")
    else:
        if args.clean_output_dir:
            clean_output_dir(output_dir)
            print(f"  Cleaned output directory: {output_dir}")
        materialize_images(
            selected_rows,
            output_dir=output_dir,
            link_mode=args.link_mode,
        )
        write_ids_file(ids_file, selected_rows)
        write_manifest(manifest_path, selected_rows)
        write_summary(summary_path, selected_rows, attribute=args.attribute)
        save_scores_cache(selected_rows, score_cache_path)

    by_city: dict[str, int] = defaultdict(int)
    for row in selected_rows:
        by_city[str(row["city"])] += 1

    print("\n-- Summary ------------------------------------------------------------")
    print(f"  Selected size: {len(selected_rows)}")
    print(f"  Cities:        {dict(sorted(by_city.items()))}")
    print(f"  Attribute:     {args.attribute}")
    print(f"  Output dir:    {output_dir}")
    print(f"  IDs file:      {ids_file}")
    print(f"  Manifest:      {manifest_path}")
    print(f"  Summary:       {summary_path}")
    print(f"  Score cache:   {score_cache_path}")
    print(
        "\n  Next: uv run python -m scripts.generate_counterfactual "
        f"--input-dir {output_dir} --input-ids {ids_file}\n"
    )


if __name__ == "__main__":
    main()
