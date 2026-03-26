from __future__ import annotations

"""
Exploratory auxiliary scoring for accepted lever edits.

This module keeps classifier scoring separate from the paper's primary
human-evaluation endpoint. It measures:

    delta_i = f_a(x'_i) - f_a(x)

for accepted edits, using cached baseline scores when available.
An exploratory auxiliary threshold ``theta`` may be supplied to mark
candidate rows and per-image summaries, but this is explicitly not the
paper's final human-grounded definition of E(x, a).
"""

import csv
import json
import os
import shutil
import statistics
import sys
from html import escape
from pathlib import Path
from typing import Any

from src.lever_identity import (
    lever_identity_label,
    serialize_identity_labels,
    serialize_identity_values,
)


SCORES_CACHE = Path("data") / "specs_scores.json"
THETA_DEFAULT = 0.10
MODEL_REPO_ID = "Jiani11/human-perception-place-pulse"
MODEL_FILES = {
    "safety": "safety.pth",
    "lively": "lively.pth",
    "wealthy": "wealthy.pth",
    "beautiful": "beautiful.pth",
    "boring": "boring.pth",
    "depressing": "depressing.pth",
}


def load_baseline_scores(cache_path: str | Path | None = None) -> dict[str, float]:
    path = Path(cache_path) if cache_path is not None else SCORES_CACHE
    if not path.exists():
        raise FileNotFoundError(
            f"Baseline scores cache not found: {path}. Run scripts/prepare_specs.py first."
        )
    with path.open(encoding="utf-8") as f:
        scores = json.load(f)
    return {str(key): float(value) for key, value in scores.items() if value is not None}


def _mps_available() -> bool:
    try:
        import torch

        return bool(torch.backends.mps.is_available())
    except Exception:
        return False


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_model_dir() -> Path:
    return _project_root() / "pretrain_human_perception_classifier_pp" / "models" / "human_perception_place_pulse"


def _load_local_runtime():
    import torch
    import torch.nn as nn
    from huggingface_hub import hf_hub_download
    from PIL import Image
    from torchvision import transforms as T

    return torch, nn, hf_hub_download, Image, T


def _detect_local_device(torch, requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    if _mps_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def _download_model_dir(hf_hub_download, model_dir: Path, *, attribute: str) -> Path:
    model_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    filenames = [MODEL_FILES[attribute], "README.md"]
    for filename in filenames:
        target = model_dir / filename
        if target.exists():
            continue
        downloaded = Path(
            hf_hub_download(
                repo_id=MODEL_REPO_ID,
                filename=filename,
            )
        )
        shutil.copy2(downloaded, target)
    return model_dir


def _load_local_models(attribute: str, *, device: str = "auto"):
    if not hasattr(_load_local_models, "_cache"):
        _load_local_models._cache = {}

    model_dir = Path(os.getenv("VITPP2_MODEL_DIR", _default_model_dir()))
    cache_key = (str(model_dir), attribute, device)
    if cache_key in _load_local_models._cache:
        return _load_local_models._cache[cache_key]

    torch, nn, hf_hub_download, Image, T = _load_local_runtime()
    resolved_device = _detect_local_device(torch, requested=device)
    model_dir = _download_model_dir(hf_hub_download, model_dir, attribute=attribute)

    script_dir = _project_root() / "pretrain_human_perception_classifier_pp" / "scripts"
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    import Model_01  # noqa: F401

    model_path = model_dir / MODEL_FILES[attribute]
    try:
        model = torch.load(model_path, map_location=torch.device(resolved_device), weights_only=False)
    except TypeError:
        model = torch.load(model_path, map_location=torch.device(resolved_device))
    model = model.to(resolved_device)
    model.eval()

    transform = T.Compose(
        [
            T.Resize((384, 384)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    _load_local_models._last_device = resolved_device
    _load_local_models._cache[cache_key] = (model, transform, torch, nn, Image, resolved_device)
    return _load_local_models._cache[cache_key]


def _score_image_with_vitpp2(
    image_path: str,
    attribute: str = "safety",
    *,
    device: str = "auto",
) -> float | None:
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_API_KEY")

    if attribute in MODEL_FILES:
        try:
            model, transform, torch, nn, Image, resolved_device = _load_local_models(
                attribute,
                device=device,
            )
            image = Image.open(image_path)
            if image.mode != "RGB":
                image = image.convert("RGB")
            tensor = transform(image).unsqueeze(0).to(resolved_device)
            softmax = nn.Softmax(dim=1)
            with torch.no_grad():
                pred = model(tensor)
                return float(softmax(pred)[0][1].item() * 10.0)
        except Exception:
            pass

    if hf_token:
        try:
            import requests

            model_id = os.getenv(
                "VITPP2_HF_MODEL_ID",
                "NUS-UAL/global-streetscapes-perception",
            )
            api_url = f"https://api-inference.huggingface.co/models/{model_id}"
            headers = {"Authorization": f"Bearer {hf_token}"}
            with open(image_path, "rb") as f:
                response = requests.post(
                    api_url,
                    headers=headers,
                    data=f.read(),
                    timeout=30,
                )
            response.raise_for_status()
            preds = response.json()
            if isinstance(preds, list):
                for pred in preds:
                    if attribute.lower() in str(pred.get("label", "")).lower():
                        return float(pred["score"])
                if preds:
                    return float(preds[0].get("score", 0))
        except Exception:
            pass

    return None


def coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def extract_image_id(row: dict[str, Any]) -> str:
    input_path = row.get("input_image_path", "")
    if input_path:
        return Path(str(input_path)).stem
    return str(row.get("image_id", ""))


def load_candidate_rows(csv_path: str | Path) -> list[dict[str, Any]]:
    with Path(csv_path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def score_candidate_rows(
    rows: list[dict[str, Any]],
    *,
    attribute: str = "safety",
    baseline_scores: dict[str, float] | None = None,
    theta: float = THETA_DEFAULT,
    device: str = "auto",
) -> list[dict[str, Any]]:
    if baseline_scores is None:
        try:
            baseline_scores = load_baseline_scores()
        except FileNotFoundError:
            baseline_scores = {}

    for row in rows:
        image_id = extract_image_id(row)
        row["image_id"] = image_id
        row["auxiliary_threshold"] = round(theta, 4)
        row["baseline_score"] = None
        row["edited_score"] = None
        row["delta_classifier"] = None
        row["exceeds_auxiliary_threshold"] = False

        if not coerce_bool(row.get("critic_is_valid")):
            continue

        baseline = baseline_scores.get(image_id)
        if baseline is None:
            baseline = _score_image_with_vitpp2(
                str(row["input_image_path"]),
                attribute,
                device=device,
            )
        if baseline is None:
            continue

        output_path = row.get("output_image_path")
        if not output_path:
            continue

        edited_score = _score_image_with_vitpp2(
            str(output_path),
            attribute,
            device=device,
        )
        if edited_score is None:
            continue

        delta = round(edited_score - baseline, 4)
        row["baseline_score"] = round(baseline, 4)
        row["edited_score"] = round(edited_score, 4)
        row["delta_classifier"] = delta
        row["exceeds_auxiliary_threshold"] = delta > theta
        row["lever_identity_label"] = lever_identity_label(row)

    return rows


def compute_per_image_aux_summary(
    rows: list[dict[str, Any]],
    *,
    theta: float = THETA_DEFAULT,
) -> dict[str, dict[str, Any]]:
    by_image: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        image_id = extract_image_id(row)
        by_image.setdefault(image_id, []).append(row)

    summary: dict[str, dict[str, Any]] = {}
    for image_id, image_rows in by_image.items():
        accepted = [row for row in image_rows if coerce_bool(row.get("critic_is_valid"))]
        scored = [row for row in accepted if row.get("delta_classifier") is not None]
        effective = [
            row for row in scored if coerce_bool(row.get("exceeds_auxiliary_threshold"))
        ]
        deltas = [float(row["delta_classifier"]) for row in scored]

        baseline_score = None
        for row in scored:
            if row.get("baseline_score") is not None:
                baseline_score = float(row["baseline_score"])
                break

        summary[image_id] = {
            "image_id": image_id,
            "baseline_score": round(baseline_score, 4) if baseline_score is not None else "",
            "n_candidates": len(image_rows),
            "n_valid": len(accepted),
            "n_scored": len(scored),
            "coverage": round(len(accepted) / len(image_rows), 3) if image_rows else 0.0,
            "mean_delta_classifier": round(statistics.mean(deltas), 4) if deltas else 0.0,
            "max_delta_classifier": round(max(deltas), 4) if deltas else 0.0,
            "n_auxiliary_effective_levers": len(effective),
            "auxiliary_threshold": round(theta, 4),
            "auxiliary_effective_lever_labels": serialize_identity_labels(effective),
            "auxiliary_effective_lever_concepts": serialize_identity_values(
                effective,
                "lever_concept",
            ),
            "auxiliary_effective_scene_supports": serialize_identity_values(
                effective,
                "scene_support",
            ),
            "auxiliary_effective_intervention_directions": serialize_identity_values(
                effective,
                "intervention_direction",
            ),
            "auxiliary_effective_edit_templates": serialize_identity_values(
                effective,
                "edit_template",
            ),
            "auxiliary_effective_target_objects": serialize_identity_values(
                effective,
                "target_object",
            ),
        }
    return summary


def write_candidate_rows_csv(rows: list[dict[str, Any]], csv_path: str | Path) -> None:
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_per_image_summary_csv(summary: dict[str, dict[str, Any]], csv_path: str | Path) -> None:
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(summary.values())
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_scatter_svg(
    rows: list[dict[str, Any]],
    svg_path: str | Path,
    *,
    theta: float,
    title: str = "Baseline safety vs classifier delta",
) -> None:
    points = [
        (
            float(row["baseline_score"]),
            float(row["delta_classifier"]),
            coerce_bool(row.get("exceeds_auxiliary_threshold")),
            extract_image_id(row),
            row.get("lever_identity_label", "") or lever_identity_label(row),
        )
        for row in rows
        if row.get("baseline_score") is not None and row.get("delta_classifier") is not None
    ]
    path = Path(svg_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    width, height = 860, 560
    margin_left, margin_right, margin_top, margin_bottom = 70, 30, 50, 70
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    if not points:
        svg = (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
            f'<text x="40" y="80" font-size="20">No scored points available</text></svg>'
        )
        path.write_text(svg, encoding="utf-8")
        return

    xs = [x for x, _, _, _, _ in points]
    ys = [y for _, y, _, _, _ in points] + [theta]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    if x_min == x_max:
        x_min -= 1
        x_max += 1
    if y_min == y_max:
        y_min -= 0.1
        y_max += 0.1

    x_pad = (x_max - x_min) * 0.08
    y_pad = (y_max - y_min) * 0.15
    x_min -= x_pad
    x_max += x_pad
    y_min -= y_pad
    y_max += y_pad

    def x_px(value: float) -> float:
        return margin_left + ((value - x_min) / (x_max - x_min)) * plot_w

    def y_px(value: float) -> float:
        return margin_top + plot_h - ((value - y_min) / (y_max - y_min)) * plot_h

    theta_y = y_px(theta)

    point_markup = []
    for baseline, delta, in_e, image_id, target_object in points:
        color = "#c0392b" if in_e else "#4a5568"
        label = escape(f"{image_id}: {target_object} ({delta:+.3f})")
        point_markup.append(
            f'<circle cx="{x_px(baseline):.1f}" cy="{y_px(delta):.1f}" r="5" fill="{color}">'
            f"<title>{label}</title></circle>"
        )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">
  <rect width="100%" height="100%" fill="#fbfaf7" />
  <text x="{margin_left}" y="28" font-size="22" font-family="Helvetica, Arial, sans-serif" fill="#1f2937">{escape(title)}</text>
  <line x1="{margin_left}" y1="{margin_top + plot_h}" x2="{margin_left + plot_w}" y2="{margin_top + plot_h}" stroke="#111827" />
  <line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_h}" stroke="#111827" />
  <line x1="{margin_left}" y1="{theta_y:.1f}" x2="{margin_left + plot_w}" y2="{theta_y:.1f}" stroke="#b91c1c" stroke-dasharray="6 4" />
  <text x="{margin_left + 8}" y="{theta_y - 8:.1f}" font-size="13" fill="#b91c1c" font-family="Helvetica, Arial, sans-serif">aux threshold = {theta:.3f}</text>
  <text x="{width / 2:.1f}" y="{height - 20}" text-anchor="middle" font-size="15" font-family="Helvetica, Arial, sans-serif" fill="#111827">Baseline score</text>
  <text x="20" y="{height / 2:.1f}" transform="rotate(-90 20 {height / 2:.1f})" text-anchor="middle" font-size="15" font-family="Helvetica, Arial, sans-serif" fill="#111827">Classifier delta</text>
  {''.join(point_markup)}
  <text x="{margin_left}" y="{height - 45}" font-size="12" fill="#4b5563" font-family="Helvetica, Arial, sans-serif">Grey = scored candidate, red = delta &gt; auxiliary threshold</text>
</svg>"""
    path.write_text(svg, encoding="utf-8")


def print_aux_summary(summary: dict[str, dict[str, Any]]) -> None:
    if not summary:
        print("No per-image summaries available.")
        return

    rows = list(summary.values())
    mean_e = round(
        statistics.mean(row["n_auxiliary_effective_levers"] for row in rows),
        3,
    )
    mean_cov = round(statistics.mean(row["coverage"] for row in rows), 3)
    mean_delta = round(statistics.mean(row["mean_delta_classifier"] for row in rows), 3)
    p_multi = round(
        sum(1 for row in rows if row["n_auxiliary_effective_levers"] > 1) / len(rows),
        3,
    )

    print("---")
    print(f"mean_aux_effective: {mean_e}")
    print(f"coverage:         {mean_cov}")
    print(f"mean_delta_aux:   {mean_delta}")
    print(f"p_multi_aux:      {p_multi}")
    print(f"n_images:         {len(rows)}")


def get_last_scoring_device() -> str:
    return getattr(_load_local_models, "_last_device", "unknown")
