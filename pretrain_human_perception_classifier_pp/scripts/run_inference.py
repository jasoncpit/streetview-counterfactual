from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from huggingface_hub import snapshot_download


ATTRIBUTES = ["safety", "lively", "wealthy", "beautiful", "boring", "depressing"]
MODEL_FILES = {
    "safety": "safety.pth",
    "lively": "lively.pth",
    "wealthy": "wealthy.pth",
    "beautiful": "beautiful.pth",
    "boring": "boring.pth",
    "depressing": "depressing.pth",
}
VALID_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MODEL_REPO_ID = "Jiani11/human-perception-place-pulse"


def workspace_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_manifest_path() -> Path:
    return workspace_root() / "data" / "formatted" / "specs_image_manifest.csv"


def default_output_path() -> Path:
    return workspace_root() / "data" / "inference" / "human_perception_predictions.csv"


def default_model_dir() -> Path:
    return workspace_root() / "models" / "human_perception_place_pulse"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the human-perception-place-pulse checkpoints on a manifest or image directory."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=default_manifest_path(),
        help="CSV manifest to score. Defaults to the formatted SPECS image manifest.",
    )
    parser.add_argument(
        "--image-column",
        default="local_image_path",
        help="Column in the manifest containing image paths.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Optional image directory. If set, this overrides --manifest.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan --input-dir for images.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=default_output_path(),
        help="Output CSV path for wide predictions.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=default_model_dir(),
        help="Directory where checkpoint files should be downloaded.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda:0", "mps"],
        help="Execution device.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optionally score only the first N images after sorting.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate resolved inputs and output paths without loading models.",
    )
    return parser.parse_args()


def require_runtime_dependencies():
    missing: list[str] = []
    try:
        import torch
        import torch.nn as nn
    except ModuleNotFoundError:
        missing.append("torch")
        torch = None
        nn = None
    try:
        from PIL import Image
    except ModuleNotFoundError:
        missing.append("Pillow")
        Image = None
    try:
        from torchvision import transforms as T
    except ModuleNotFoundError:
        missing.append("torchvision")
        T = None

    if missing:
        joined = ", ".join(sorted(set(missing)))
        raise SystemExit(
            "Missing runtime dependencies: "
            f"{joined}. Install the remote repo requirements before running inference."
        )

    return torch, nn, Image, T


def detect_device(torch, requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda:0"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def read_manifest_rows(path: Path, image_column: str) -> list[dict[str, str]]:
    if not path.exists():
        raise SystemExit(f"Manifest not found: {path}")

    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return []
    if image_column not in rows[0]:
        raise SystemExit(f"Manifest column not found: {image_column}")
    return rows


def collect_input_rows(args: argparse.Namespace) -> list[dict[str, str]]:
    if args.input_dir is not None:
        if not args.input_dir.exists():
            raise SystemExit(f"Input directory not found: {args.input_dir}")
        pattern = "**/*" if args.recursive else "*"
        image_paths = sorted(
            p for p in args.input_dir.glob(pattern) if p.is_file() and p.suffix.lower() in VALID_SUFFIXES
        )
        rows = [{"image_path": str(path)} for path in image_paths]
    else:
        rows = read_manifest_rows(args.manifest, args.image_column)

    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit("No input images resolved.")
    return rows


def download_models(model_dir: Path) -> Path:
    model_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=MODEL_REPO_ID,
        local_dir=str(model_dir),
        allow_patterns=["*.pth", "README.md"],
    )
    return model_dir


def load_models(model_dir: Path, device: str, torch):
    script_dir = Path(__file__).resolve().parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    import Model_01  # noqa: F401

    models = {}
    for attribute, filename in MODEL_FILES.items():
        model_path = model_dir / filename
        try:
            model = torch.load(model_path, map_location=torch.device(device), weights_only=False)
        except TypeError:
            model = torch.load(model_path, map_location=torch.device(device))
        model = model.to(device)
        model.eval()
        models[attribute] = model
    return models


def build_transform(T):
    return T.Compose(
        [
            T.Resize((384, 384)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def predict_scores(image_path: Path, models, transform, device: str, torch, nn, Image) -> dict[str, float]:
    image = Image.open(image_path)
    if image.mode != "RGB":
        image = image.convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    softmax = nn.Softmax(dim=1)

    predictions: dict[str, float] = {}
    with torch.no_grad():
        for attribute, model in models.items():
            pred = model(tensor)
            score = softmax(pred)[0][1].item() * 10.0
            predictions[f"pred_{attribute}"] = round(score, 4)
    return predictions


def resolve_image_path(row: dict[str, str], args: argparse.Namespace) -> Path:
    if args.input_dir is not None:
        return Path(row["image_path"])
    return Path(row[args.image_column])


def write_predictions(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    input_rows = collect_input_rows(args)

    if args.dry_run:
        print(f"resolved_rows={len(input_rows)}")
        print(f"output_csv={args.output_csv}")
        if args.input_dir is None:
            print(f"manifest={args.manifest}")
            print(f"image_column={args.image_column}")
        else:
            print(f"input_dir={args.input_dir}")
        return

    torch, nn, Image, T = require_runtime_dependencies()
    device = detect_device(torch, args.device)
    model_dir = download_models(args.model_dir)
    models = load_models(model_dir, device, torch)
    transform = build_transform(T)

    output_rows: list[dict[str, str | float]] = []
    for row in input_rows:
        image_path = resolve_image_path(row, args)
        if not image_path.exists():
            raise SystemExit(f"Image not found: {image_path}")
        scored_row: dict[str, str | float] = dict(row)
        scored_row["resolved_image_path"] = str(image_path)
        scored_row.update(predict_scores(image_path, models, transform, device, torch, nn, Image))
        output_rows.append(scored_row)

    fieldnames = list(output_rows[0].keys())
    write_predictions(args.output_csv, output_rows, fieldnames)

    print(f"device={device}")
    print(f"model_dir={model_dir}")
    print(f"rows_scored={len(output_rows)}")
    print(f"output_csv={args.output_csv}")


if __name__ == "__main__":
    main()
