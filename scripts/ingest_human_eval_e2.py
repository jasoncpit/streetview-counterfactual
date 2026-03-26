from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest E2 human evaluation responses and compute final paper-endpoint summaries.",
    )
    parser.add_argument("--manifest", required=True, help="Manifest CSV from export_human_eval_e2.py")
    parser.add_argument("--responses", required=True, help="CSV with one row per rater response")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.65,
        help="Preference-share threshold for final human effectiveness.",
    )
    parser.add_argument(
        "--pair-output",
        default=None,
        help="Optional output CSV path for pair-level aggregated results.",
    )
    parser.add_argument(
        "--image-output",
        default=None,
        help="Optional output CSV path for per-image final endpoint summaries.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return [dict(row) for row in csv.DictReader(f)]


def write_csv(path: Path, rows: list[dict[str, str | int | float | bool]], fieldnames: list[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def selected_role(response_row: dict[str, str], manifest_row: dict[str, str]) -> str:
    raw = (
        response_row.get("selected_side")
        or response_row.get("response_side")
        or response_row.get("choice")
        or ""
    ).strip().lower()
    if raw in {"left", "right"}:
        return manifest_row[f"{raw}_image_role"]
    if raw in {"original", "edited"}:
        return raw
    raise ValueError(
        f"Unsupported response value '{raw}' for pair_id={response_row.get('pair_id', '')}."
    )


def aggregate_pair_results(
    manifest_rows: list[dict[str, str]],
    response_rows: list[dict[str, str]],
    *,
    threshold: float,
) -> list[dict[str, str | int | float | bool]]:
    manifest_by_pair = {row["pair_id"]: row for row in manifest_rows}
    grouped: dict[str, list[dict[str, str]]] = {row["pair_id"]: [] for row in manifest_rows}
    for response in response_rows:
        pair_id = str(response.get("pair_id", "")).strip()
        if not pair_id or pair_id not in manifest_by_pair:
            continue
        grouped[pair_id].append(response)

    pair_rows: list[dict[str, str | int | float | bool]] = []
    for pair_id, manifest in manifest_by_pair.items():
        responses = grouped.get(pair_id, [])
        edited_votes = sum(
            1 for response in responses if selected_role(response, manifest) == "edited"
        )
        n_ratings = len(responses)
        preference_share = round(edited_votes / n_ratings, 4) if n_ratings else 0.0
        pair_rows.append(
            {
                "pair_id": pair_id,
                "image_id": manifest["image_id"],
                "candidate_id": manifest["candidate_id"],
                "target_attribute": manifest["target_attribute"],
                "lever_identity_label": manifest["lever_identity_label"],
                "lever_concept": manifest["lever_concept"],
                "scene_support": manifest["scene_support"],
                "intervention_direction": manifest["intervention_direction"],
                "edit_template": manifest["edit_template"],
                "planner_target_object": manifest["planner_target_object"],
                "planner_edit_plan": manifest["planner_edit_plan"],
                "n_ratings": n_ratings,
                "edited_votes": edited_votes,
                "edited_preference_share": preference_share,
                "human_effect_threshold": round(threshold, 4),
                "is_human_effective": n_ratings > 0 and preference_share >= threshold,
            }
        )
    return pair_rows


def summarize_by_image(
    pair_rows: list[dict[str, str | int | float | bool]],
) -> list[dict[str, str | int | float]]:
    grouped: dict[str, list[dict[str, str | int | float | bool]]] = {}
    for row in pair_rows:
        grouped.setdefault(str(row["image_id"]), []).append(row)

    image_rows: list[dict[str, str | int | float]] = []
    for image_id, rows in grouped.items():
        effective = [row for row in rows if bool(row["is_human_effective"])]
        rated = [row for row in rows if int(row["n_ratings"]) > 0]
        mean_share = (
            round(statistics.mean(float(row["edited_preference_share"]) for row in rated), 4)
            if rated
            else 0.0
        )
        image_rows.append(
            {
                "image_id": image_id,
                "n_tested_accepted_levers": len(rows),
                "n_rated_pairs": len(rated),
                "n_human_effective_levers": len(effective),
                "mean_edited_preference_share": mean_share,
                "human_effective_lever_labels": "|".join(
                    str(row["lever_identity_label"]) for row in effective
                ),
            }
        )
    return image_rows


def default_output_path(path: Path, suffix: str) -> Path:
    return path.with_name(f"{path.stem}{suffix}{path.suffix}")


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest)
    responses_path = Path(args.responses)
    manifest_rows = read_csv(manifest_path)
    response_rows = read_csv(responses_path)
    pair_rows = aggregate_pair_results(
        manifest_rows,
        response_rows,
        threshold=args.threshold,
    )
    image_rows = summarize_by_image(pair_rows)

    pair_output = (
        Path(args.pair_output)
        if args.pair_output
        else default_output_path(responses_path, "_pair_results")
    )
    image_output = (
        Path(args.image_output)
        if args.image_output
        else default_output_path(responses_path, "_image_summary")
    )
    write_csv(
        pair_output,
        pair_rows,
        [
            "pair_id",
            "image_id",
            "candidate_id",
            "target_attribute",
            "lever_identity_label",
            "lever_concept",
            "scene_support",
            "intervention_direction",
            "edit_template",
            "planner_target_object",
            "planner_edit_plan",
            "n_ratings",
            "edited_votes",
            "edited_preference_share",
            "human_effect_threshold",
            "is_human_effective",
        ],
    )
    write_csv(
        image_output,
        image_rows,
        [
            "image_id",
            "n_tested_accepted_levers",
            "n_rated_pairs",
            "n_human_effective_levers",
            "mean_edited_preference_share",
            "human_effective_lever_labels",
        ],
    )

    print(f"Wrote pair-level E2 results: {pair_output}")
    print(f"Wrote image-level E2 summary: {image_output}")


if __name__ == "__main__":
    main()
