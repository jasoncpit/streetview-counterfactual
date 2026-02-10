Counterfactual-StreetView
=========================

Abstract
--------
Street-view perception models can predict subjective attributes such as safety, wealth, or boringness, yet they rarely answer the question urban planning practitioners care about: for a particular street, what visual evidence drives the judgment, and what localized change would shift perceived safety without altering unrelated cues? We propose an interventional approach to interpretability that treats counterfactual edits as *mechanism probes* for human perception. Given an image and a target attribute, we produce a testable explanation consisting of (i) a target object/region *t*, (ii) an evidence description *e*, and (iii) a generated image *x'* intended to differ from *x* only in the factor described by *e*. Our evaluation is intentionally *generation-model agnostic*: *x'* can be produced by prompt-only generators, inpainting/editing models, or proprietary systems. Rather than assuming edits are faithful, we formalize a human-judgment protocol that measures validity (same-place, locality, realism) and directional perception shift. We then estimate counterfactual effects using randomized pairwise human judgments (edited vs. original), producing per-street effect sizes with uncertainty and explicit failure diagnostics when faithful edits cannot be delivered. This reframes interpretability for urban perception from correlational saliency and narratives to testable evidence grounded in human feedback.

High-level description
----------------------
The pipeline takes a street-level image and a target perceptual attribute (e.g. "safety"), then:

1. **Plan** — an LLM planner proposes a single, minimal, localized visual edit and a target object.
2. **Edit** — an image-editing model applies the edit to the original image.
3. **Critique** — an LLM critic compares the original and edited images for realism and minimality; if the edit fails, the loop retries with planner feedback.

The result is a counterfactual image pair (original vs. edited) and structured metadata (edit plan, target object, critic verdict) suitable for downstream human evaluation.

Repository structure
--------------------
```
streetview-counterfactual/
├── scripts/
│   └── generate_counterfactual.py   # Batch generation over all images → CSV
├── src/
│   ├── config.py                    # All dataclass configs (project, workflow, agents, logging)
│   ├── integrations/
│   │   ├── openai_client.py         # OpenAI planner & critic (GPT vision)
│   │   └── replicate_client.py      # Replicate API wrapper (image editing models)
│   ├── utils/
│   │   ├── logging.py               # Logging setup
│   │   ├── paths.py                 # ensure_dir, timestamped_path helpers
│   │   └── pipeline.py              # collect_images, run_baseline_for_image
│   └── workflow/
│       ├── graph.py                 # LangGraph workflow (baseline & full pipelines)
│       ├── state.py                 # AgentState TypedDict
│       └── nodes/
│           ├── planning.py          # plan_edit_node
│           ├── criticism.py         # critique_generated_node
│           ├── generation.py        # inpaint_node
│           └── segmentation.py      # segment_object_node
├── tests/
│   └── test_replicate_client.py
├── data/
│   ├── 01_raw/                      # Input street-level images
│   ├── 02_counterfactual/           # Edited counterfactual images
│   └── 03_eval_results/             # CSV evaluation outputs
├── notebooks/
│   └── pipeline_walkthrough.ipynb   # Interactive walkthrough
├── pyproject.toml
└── README.md
```

Setup
-----
1. Clone the repo and install dependencies:
   ```bash
   pip install -e .
   ```
2. Create a `.env` file in the project root with your API keys:
   ```
   OPENAI_API_KEY=sk-...
   REPLICATE_API_TOKEN=r8_...
   ```
3. Place input images (`.png`, `.jpg`, `.jpeg`, `.webp`) in `data/01_raw/`.

Generating counterfactuals
--------------------------
Run `scripts/generate_counterfactual.py` to process all images in the input directory and produce a CSV summary:

```bash
python scripts/generate_counterfactual.py
```

### CLI options

| Flag | Default | Description |
|---|---|---|
| `--model` | `black-forest-labs/flux-kontext-max` | Replicate model slug for image editing |
| `--max-attempts` | `1` | Max plan → edit → critic retries per image |
| `--target-attribute` | `safety` | Perceptual attribute to shift (e.g. safety, wealth, greenery) |
| `--input-dir` | `data/01_raw` | Override the input image directory |
| `--csv-path` | auto-timestamped in `data/03_eval_results/` | Override the output CSV path |

### Examples

```bash
# Default: flux-kontext-max, 1 attempt, safety
python scripts/generate_counterfactual.py

# Use a different model with 3 retries
python scripts/generate_counterfactual.py --model google/nano-banana-pro --max-attempts 3

# Target a different attribute
python scripts/generate_counterfactual.py --target-attribute wealth

# Custom input and output paths
python scripts/generate_counterfactual.py --input-dir /path/to/images --csv-path results.csv
```

The output CSV contains one row per image with columns: `input_image_path`, `output_image_path`, `planner_edit_plan`, `planner_target_object`, `critic_is_realistic`, `critic_is_minimal_edit`, `critic_notes`.

Model configuration
-------------------
All defaults live in `src/config.py` as plain Python dataclasses. Key settings:

| Config | Field | Default |
|---|---|---|
| `WorkflowConfig` | `baseline_model` | `black-forest-labs/flux-kontext-max` |
| `WorkflowConfig` | `openai_model` | `gpt-5.2` |
| `WorkflowConfig` | `target_attribute` | `safety` |
| `WorkflowConfig` | `max_attempts` | `3` |

### Supported baseline edit models

The following models can be passed via `--model`:

- `black-forest-labs/flux-kontext-max` — FLUX Kontext Max (default)
- `google/nano-banana-pro` — Nano Banana Pro
- `bytedance/seedream-4` — SeedReam 4
- `openai/gpt-image-1.5` — GPT Image 1.5
- `qwen/qwen-image-edit` — Qwen Image Edit

To change the default model, edit `baseline_model` in `WorkflowConfig` inside `src/config.py`.
