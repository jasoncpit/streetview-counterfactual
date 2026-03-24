# pretrain_human_perception_classifier_pp

Local workspace for preparing pretraining data for the `strawmelon11/human-perception-place-pulse` classifier family without cloning that repo.

## What is here

- `docs/remote_repo_inspection.md`: concise inspection of the remote GitHub repo.
- `scripts/download_and_format_specs.py`: downloads `matiasqr/specs` from Hugging Face and writes normalized CSV manifests.
- `data/raw/`: raw Hugging Face snapshot location.
- `data/formatted/`: normalized training-ready tables.

## Usage

From the repository root:

```bash
python pretrain_human_perception_classifier_pp/scripts/download_and_format_specs.py
```

This will:

1. Download the `matiasqr/specs` dataset snapshot into `pretrain_human_perception_classifier_pp/data/raw/specs/`.
2. Build a wide per-image manifest with image paths, metadata, visual complexity, inferred perception scores, and Q-scores.
3. Build a normalized pairwise-comparison table from the raw human judgments.
4. Write a small `dataset_summary.json` file with counts.

## Inference

To score the formatted SPECS manifest with the published checkpoints:

```bash
uv run --python 3.12 --with torch --with torchvision --with huggingface_hub --with pillow \
  python pretrain_human_perception_classifier_pp/scripts/run_inference.py
```

Useful variants:

```bash
# Only validate resolved inputs and output locations
python pretrain_human_perception_classifier_pp/scripts/run_inference.py --dry-run

# Score a custom folder of images instead of the manifest
python pretrain_human_perception_classifier_pp/scripts/run_inference.py \
  --input-dir /path/to/images \
  --recursive \
  --output-csv pretrain_human_perception_classifier_pp/data/inference/custom_predictions.csv
```

The inference script downloads the six published `.pth` checkpoints into
`pretrain_human_perception_classifier_pp/models/human_perception_place_pulse/`
and writes one wide CSV with `pred_safety`, `pred_lively`, `pred_wealthy`,
`pred_beautiful`, `pred_boring`, and `pred_depressing`.

For the main repo's auxiliary scoring pipeline, you can also download a
single checkpoint directly with:

```bash
./scripts/download_vitpp2.sh --attribute safety
```

This is useful when running `scripts/run_analysis.py` locally on Apple
Silicon with `--device mps`.
