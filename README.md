Counterfactual-StreetView
=========================

Goal
----
Probe whether vision-language models remain causally aligned with human perception by editing street-level imagery along mid-level visual attributes (e.g., lighting, trees, enclosure) and evaluating the resulting percepts.

Pipeline (high level)
---------------------
- Hydra configs live in `configs/` (`main.yaml`, `agents.yaml`, `tools.yaml`).
- LangGraph orchestrates the workflow in `src/workflow/graph.py`, passing `AgentState` between nodes.
- External APIs are wrapped in `src/integrations/` (OpenAI for planning, Replicate for DINO/SAM-3/Nano Banana/inpainting).
- Data is organized under `data/` (raw images, masks, counterfactuals, evaluation outputs).
- Scripts in `scripts/` bootstrap the pipeline and post-hoc analysis.

Quickstart
----------
1) Create a `.env` with `OPENAI_API_KEY` and `REPLICATE_API_TOKEN`.
2) Install dependencies: `pip install -e .`
3) Run the pipeline (example): `python scripts/run_pipeline.py target_attribute="safety" input_dir="data/01_raw"`

Notes
-----
- Segmentation is implemented as a **Grounded-SAM-2 style** pipeline: Grounding DINO (text → box) followed by SAM 2 (box → mask). The reference we follow is [IDEA-Research/Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2/tree/main).
- The current implementation scaffolds the LangGraph workflow and API wrappers with placeholders for model IDs and prompts. Swap in your Replicate model hashes and refine prompts as you iterate. Public slugs observed on Replicate’s catalog you can try (verify access in your account): `adirik/grounding-dino:latest` for detection, `meta/sam-2:latest` for masks, `black-forest-labs/flux-fill-pro:latest` for inpainting, and the VLM scorer `google/nano-banana-pro:latest` (or any realism scorer you have access to). Replace these in `configs/tools/default.yaml` if different in your workspace. See model pages for details: [Grounding DINO](https://replicate.com/adirik/grounding-dino), [FLUX Fill Pro](https://replicate.com/black-forest-labs/flux-fill-pro).
- The current implementation scaffolds the LangGraph workflow and API wrappers with placeholders for model IDs and prompts. Swap in your Replicate model hashes and refine prompts as you iterate. Public slugs observed on Replicate’s catalog you can try (verify access in your account): `adirik/grounding-dino:latest` for detection, `meta/sam-2:latest` for masks, `black-forest-labs/flux-fill-pro:latest` for inpainting (supports extra params like `steps`, `guidance`, `output_format` via `inpaint_params` in `configs/tools/default.yaml`), and the VLM scorer `google/nano-banana-pro:latest` (or any realism scorer you have access to). Replace these in `configs/tools/default.yaml` if different in your workspace. See model pages for details: [Grounding DINO](https://replicate.com/adirik/grounding-dino), [FLUX Fill Pro](https://replicate.com/black-forest-labs/flux-fill-pro).

Test status (latest run)
------------------------
- Latest run (2025-12-11 21:30 UTC): `PYTHONPATH=$PWD UV_CACHE_DIR=.uvcache uv run python scripts/run_pipeline.py workflow.target_attribute="safety" workflow.input_dir="data/01_raw"`.
- Inputs: `data/01_raw/image.png` (user-provided), plus earlier 1x1 placeholders (`sample_red.png`, `sample_blue.png`).
- Replicate responses: `idea-research/ram-grounded-sam:latest` segmentation returned 422 (invalid/not permitted) and 429 throttles; inpaint `stability-ai/stable-diffusion-inpainting:latest` hit 422/429; realism `google/nano-banana-pro:latest` hit 429. Client fell back to mock masks/edits and realism=0.5.
- Outputs: masks in `data/02_masks/` and counterfactuals in `data/03_counterfactuals/` for each of 3 attempts (fallback copies when Replicate failed). Critic marked all attempts unrealistic (vlm fallback 0.5 + LLM critique).
- OpenAI calls succeeded; loop stopped after `max_attempts=3` with plans targeting streetlights/crosswalks.

Logging improvements
--------------------
- Replicate client now logs start/end of segmentation, inpaint, and realism scoring with the model IDs and prompts used, plus verbose warnings (with stack traces) when falling back to mocks. Configure verbosity via `configure_logging` in `src/utils/logging.py` or pass `LOG_LEVEL` env var before running.

Next steps to run the pipeline
------------------------------
- Confirm the Replicate model slugs/versions you have access to (422 indicates invalid/unauthorized; 429 indicates rate limiting). Update `configs/tools/default.yaml` with permitted hashes or increase credit to avoid throttling.
- Re-run with uv: `PYTHONPATH=$PWD UV_CACHE_DIR=.uvcache uv run python scripts/run_pipeline.py workflow.target_attribute="safety" workflow.input_dir="data/01_raw"`.
- Ensure `.env` has `OPENAI_API_KEY` and `REPLICATE_API_TOKEN`; expect charges when real models are used.
- If you want strict realism gating once models succeed, raise `workflow.realism_threshold`; increase `workflow.max_attempts` for more retries.
