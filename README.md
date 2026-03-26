Counterfactual-StreetView
=========================

Overview
--------
This repo implements a paper-aligned pilot for street-view interventional attribution.
For each image, the system:

1. constructs a bounded scene-specific set of lever candidates `l = (c, s, d, τ)`,
2. realizes each candidate with prompt-only image editing,
3. audits each result for same-place preservation, locality, realism, and plausibility,
4. exports candidate-level outputs for downstream auxiliary scoring and optional future human evaluation.

The current codebase supports:

- closed-ontology lever candidate generation,
- prompt-only editing with `flux-kontext-max`,
- four-part validity auditing,
- auxiliary ViT-PP2 scoring with a local checkpoint,
- per-image auxiliary summaries and scatter plots,
- optional E2 human-evaluation export and ingestion for future studies.

The current paper-facing path reports validity and auxiliary classifier diagnostics. Human evaluation remains optional and is not required to reproduce the current pilot.

Environment Setup
-----------------
Install dependencies with `uv`:

```bash
uv sync
```

Create a local `.env` file with the required API credentials:

```bash
OPENAI_API_KEY=...
REPLICATE_API_TOKEN=...
```

The repo assumes Python 3.12+ and resolves most scripts relative to the repository root. Run commands from the top-level project directory.

If you plan to run auxiliary classifier scoring locally, download the checkpoint once:

```bash
./scripts/download_vitpp2.sh --attribute safety
```

End-to-End Workflow
-------------------
The supported workflow is:

1. prepare a paper image subset,
2. generate bounded candidate edits for each image,
3. run auxiliary classifier scoring if needed for analysis or prioritization,
4. optionally export accepted edits for E2 human evaluation,
5. optionally ingest completed human judgments for a future human-grounded study.

For a reproducible pilot run, use the single-driver script:

```bash
uv run python scripts/run_repro_pipeline.py \
  --prefix specs_repro_n20 \
  --n-total 20 \
  --candidate-budget 3 \
  --max-attempts 1 \
  --score-device mps
```

This command:

- materializes a deterministic `n=20` subset with run-specific manifests and score cache,
- resumes generation from `data/03_eval_results/specs_repro_n20.csv` if a previous run was interrupted,
- runs auxiliary scoring on the same subset,
- stops after subset preparation, generation, and auxiliary scoring.

If you also want the optional E2 export package, add `--export-human-eval`.

Run-specific artifacts are written under:

- `data/01_raw/specs_repro_n20/`
- `data/specs_repro_n20_config.json`
- `data/specs_repro_n20_ids.txt`
- `data/specs_repro_n20_manifest.csv`
- `data/specs_repro_n20_summary.json`
- `data/specs_repro_n20_scores.json`
- `data/03_eval_results/specs_repro_n20.csv`
- `data/03_eval_results/specs_repro_n20_scored.csv`
- `data/03_eval_results/specs_repro_n20_per_image_auxiliary.csv`
- `data/03_eval_results/specs_repro_n20_baseline_vs_aux_delta.svg`

Step 1: Prepare the paper image subset
--------------------------------------
Prepare the paper-facing SPECS subset:

```bash
uv run python scripts/prepare_specs.py --clean-output-dir
```

By default this creates a balanced `N=100` subset:

- 20 images from each of the 5 available SPECS cities,
- stratified within city by inferred safety score and visual complexity,
- materialized into `data/01_raw/specs_paper_n100/`,
- listed in `paper_ids.txt`,
- summarized in `data/specs_paper_n100_manifest.csv`.

If you want a smaller exploratory subset, you can still override the defaults, for example:

```bash
uv run python scripts/prepare_specs.py --city singapore --n-total 30 --clean-output-dir
```

Step 2: Generate counterfactual candidates
------------------------------------------
Generate counterfactual candidates:

```bash
uv run python -m scripts.generate_counterfactual \
  --input-dir data/01_raw/specs_paper_n100 \
  --input-ids paper_ids.txt \
  --candidate-budget 3 \
  --max-attempts 1 \
  --target-attribute safety \
  --csv-path data/03_eval_results/specs_paper_n100.csv
```

Important generation semantics:

- Each output row corresponds to one candidate lever for one image.
- `--candidate-budget` controls how many candidate levers the planner may return per image.
- `--max-attempts` is the bounded stochastic attempt budget for each fixed candidate lever.
- Planner outputs are filtered in code so `lever_concept` must match the configured ontology exactly after normalization.
- Invalid ontology members are dropped rather than coerced to a nearest ontology label.
- Retry semantics are explicit: repeated attempts correspond to repeated stochastic generations of the same candidate lever, not hidden replanning.

Useful generation options:

- `--input-path`: run a single image directly.
- `--input-ids`: run only a selected list of image IDs from the input directory.
- `--input-dir`: override the raw image directory.
- `--csv-path`: control where the candidate-level CSV is written.
- `--resume`: skip images that already have rows in the output CSV and continue from the last checkpoint.

Step 3: Run auxiliary classifier scoring
---------------------------------------
Run auxiliary scoring on Apple Silicon with `mps`:

```bash
PYTHONPATH=. uv run --isolated --no-project --python 3.12 \
  --with torch --with torchvision --with huggingface_hub --with pillow --with rich \
  python -m scripts.run_analysis \
  --csv data/03_eval_results/specs_paper_n100.csv \
  --attribute safety \
  --theta 0.10 \
  --device mps
```

Notes:

- The auxiliary scorer uses ViT-PP2 to compute `delta_classifier = f_a(x') - f_a(x)` for accepted edits only.
- The threshold passed with `--theta` is auxiliary only. It is not the final human endpoint.
- The scored candidate CSV and per-image summary CSV are written next to the input CSV by default.
- The scatter SVG visualizes baseline classifier score versus auxiliary classifier delta.

Optional Step 4: Export accepted edits for E2 pairwise human evaluation
----------------------------------------------------------------------
Export accepted edits for E2 pairwise human evaluation:

```bash
uv run python -m scripts.export_human_eval_e2 \
  --csv data/03_eval_results/specs_paper_n100.csv
```

This creates a package under `data/06_human_eval_exports/<csv_stem>/` containing:

- `e2_manifest.csv`: one row per accepted candidate pair,
- `e2_responses_template.csv`: a blank response template,
- `assets/`: copied original and edited images used by the study.

The export randomizes whether the original or edited image appears on the left or right for each pair and stores that mapping in the manifest.

Optional Step 5: Ingest completed E2 responses
---------------------------------------------
Ingest completed E2 responses:

```bash
uv run python -m scripts.ingest_human_eval_e2 \
  --manifest data/06_human_eval_exports/specs_paper_n100/e2_manifest.csv \
  --responses data/06_human_eval_exports/specs_paper_n100/e2_responses_template.csv
```

The ingestion script expects one response row per human judgment with:

- `pair_id`
- `rater_id`
- `selected_side`

`selected_side` may be `left`, `right`, `original`, or `edited`.

The script writes:

- pair-level aggregated E2 results with edited-preference share,
- image-level final summaries counting human-effective levers.

Core Outputs
------------
Generation produces one row per candidate lever, not one row per image.

Candidate CSV fields include:

- `candidate_id`
- `target_attribute`
- `lever_concept`
- `scene_support`
- `intervention_direction`
- `edit_template`
- `planner_edit_plan`
- `planner_target_object`
- `lever_identity_label`
- `critic_same_place_preserved`
- `critic_is_localized`
- `critic_is_realistic`
- `critic_is_plausible`
- `critic_is_valid`
- `stochastic_attempt_budget`
- `stochastic_attempts_used`

Candidate field meanings:

- `scene_support`: short free-text grounding phrase naming the visible support in the scene.
- `planner_target_object`: short noun phrase for the object being edited.
- `lever_identity_label`: canonical human-readable identity string built from the full lever specification.
- `stochastic_attempt_budget`: maximum bounded stochastic attempts allowed for that candidate.
- `stochastic_attempts_used`: realized attempts before success or exhaustion.

Auxiliary scoring adds:

- `baseline_score`
- `edited_score`
- `delta_classifier`
- `auxiliary_threshold`
- `exceeds_auxiliary_threshold`

Per-image auxiliary summaries include:

- `n_candidates`
- `n_valid`
- `n_scored`
- `coverage`
- `mean_delta_classifier`
- `max_delta_classifier`
- `n_auxiliary_effective_levers`
- `auxiliary_effective_lever_labels`
- `auxiliary_effective_lever_concepts`
- `auxiliary_effective_scene_supports`
- `auxiliary_effective_intervention_directions`
- `auxiliary_effective_edit_templates`
- `auxiliary_effective_target_objects`

These summary fields are deliberately named as auxiliary to avoid confusion with the paper's human-defined `E(x,a)`.

Human E2 outputs include:

- `pair_id`
- `question_text`
- `left_image_role`
- `right_image_role`
- `edited_preference_share`
- `is_human_effective`

Repository Semantics
--------------------
- The only supported generation path is the paper-facing multi-candidate pipeline in [`scripts/generate_counterfactual.py`](/Users/jason_macstudio/streetview-counterfactual/scripts/generate_counterfactual.py).
- Legacy graph-based workflow code has been removed rather than maintained in parallel.
- `flux-kontext-max` is the current generator implementation, but the paper framing remains generator-agnostic.
- For the current draft, auxiliary classifier scoring is the reported pilot analysis. Human evaluation remains an optional extension.

Repository Structure
--------------------
```text
streetview-counterfactual/
├── scripts/
│   ├── prepare_specs.py
│   ├── generate_counterfactual.py
│   ├── run_analysis.py
│   ├── export_human_eval_e2.py
│   ├── ingest_human_eval_e2.py
│   ├── download_vitpp2.sh
│   └── build_evidence_pack.py
├── src/
│   ├── config.py
│   ├── lever_identity.py
│   ├── scoring.py
│   ├── integrations/
│   │   ├── openai_client.py
│   │   └── replicate_client.py
│   ├── utils/
│   │   ├── logging.py
│   │   ├── paths.py
│   │   └── pipeline.py
│   └── workflow/
│       └── state.py
├── paper/
├── pretrain_human_perception_classifier_pp/
├── tests/
└── data/
```

Notes
-----
- The paper-facing method is multi-candidate and validity-gated. Legacy graph-based single-edit workflow code has been removed.
- The local ViT-PP2 scoring path prefers a local checkpoint and supports `--device mps` on Apple Silicon.
- Human-defined `E(x,a)` is computed only from the E2 human-evaluation ingestion path. Auxiliary classifier summaries are exploratory only.
- Existing generated data under `data/` may still use older field names from earlier runs. Re-run the pipeline to regenerate outputs with the current contracts.
