Counterfactual-StreetView
=========================

Overview
--------
This repo implements a paper-aligned pilot for street-view interventional attribution.
For each image, the system:

1. constructs a bounded scene-specific set of lever candidates `l = (c, s, d, τ)`,
2. realizes each candidate with prompt-only image editing,
3. audits each result for same-place preservation, locality, realism, and plausibility,
4. exports candidate-level outputs for downstream auxiliary scoring and future human evaluation.

The current codebase supports:

- closed-ontology lever candidate generation,
- prompt-only editing with `flux-kontext-max`,
- four-part validity auditing,
- auxiliary ViT-PP2 scoring with a local checkpoint,
- per-image auxiliary summaries and scatter plots.

The primary paper endpoint remains human evaluation. The classifier analysis in this repo is auxiliary only.

Quick Start
-----------
Install dependencies with `uv`:

```bash
uv sync
```

Create a local `.env` file with the required API credentials:

```bash
OPENAI_API_KEY=...
REPLICATE_API_TOKEN=...
```

Prepare a small Singapore pilot from SPECS:

```bash
uv run python scripts/prepare_specs.py --n-per-stratum 1 --city singapore --attribute safe
```

Generate counterfactual candidates:

```bash
uv run python -m scripts.generate_counterfactual \
  --input-ids pilot_ids.txt \
  --candidate-budget 3 \
  --max-attempts 1 \
  --target-attribute safety \
  --csv-path data/03_eval_results/singapore_pilot_3x3.csv
```

Download the local safety checkpoint once:

```bash
./scripts/download_vitpp2.sh --attribute safety
```

Run auxiliary scoring on Apple Silicon with `mps`:

```bash
PYTHONPATH=. uv run --isolated --no-project --python 3.12 \
  --with torch --with torchvision --with huggingface_hub --with pillow --with rich \
  python -m scripts.run_analysis \
  --csv data/03_eval_results/singapore_pilot_3x3.csv \
  --attribute safety \
  --theta 0.10 \
  --device mps
```

Core Outputs
------------
Generation produces one row per candidate lever, not one row per image.

Candidate CSV fields include:

- `lever_concept`
- `scene_support`
- `intervention_direction`
- `edit_template`
- `planner_edit_plan`
- `planner_target_object`
- `critic_same_place_preserved`
- `critic_is_localized`
- `critic_is_realistic`
- `critic_is_plausible`
- `critic_is_valid`
- `attempts_used`

Auxiliary scoring adds:

- `baseline_score`
- `edited_score`
- `delta_classifier`
- `theta_aux`
- `in_E_aux`

Per-image auxiliary summaries include:

- `n_candidates`
- `n_valid`
- `n_scored`
- `coverage`
- `mean_delta_classifier`
- `max_delta_classifier`
- `E_size_aux`
- `effective_levers_aux`

Repository Structure
--------------------
```text
streetview-counterfactual/
├── scripts/
│   ├── prepare_specs.py
│   ├── generate_counterfactual.py
│   ├── run_analysis.py
│   ├── download_vitpp2.sh
│   └── build_evidence_pack.py
├── src/
│   ├── config.py
│   ├── scoring.py
│   ├── integrations/
│   │   ├── openai_client.py
│   │   └── replicate_client.py
│   ├── utils/
│   │   ├── logging.py
│   │   ├── paths.py
│   │   └── pipeline.py
│   └── workflow/
│       ├── graph.py
│       ├── state.py
│       └── nodes/
├── paper/
├── pretrain_human_perception_classifier_pp/
├── tests/
└── data/
```

Notes
-----
- The paper-facing method is multi-candidate and validity-gated. Old single-edit descriptions are obsolete.
- The local ViT-PP2 scoring path prefers a local checkpoint and supports `--device mps` on Apple Silicon.
- Human-defined `E(x,a)` is deferred; `E_size_aux` is exploratory only.
