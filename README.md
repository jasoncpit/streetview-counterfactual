Counterfactual-StreetView
=========================

Goal
----
Probe whether vision-language models remain causally aligned with human perception by editing street-level imagery along mid-level visual attributes (e.g., lighting, trees, enclosure) and evaluating the resulting percepts.

Pipeline (high level)
---------------------
- Plain Python config in `src/config.py`.
- LangGraph orchestrates the workflow in `src/workflow/graph.py`, passing `AgentState` between nodes.
- External APIs are wrapped in `src/integrations/` (OpenAI for planning, Replicate for DINO/SAM-3/Nano Banana/inpainting).
- Data is organized under `data/` (raw images, masks, counterfactuals, evaluation outputs).
- Scripts in `scripts/` bootstrap the pipeline and post-hoc analysis.

Quickstart
----------
1) Create a `.env` with `OPENAI_API_KEY` and `REPLICATE_API_TOKEN`.
2) Install dependencies: `pip install -e .`
3) Run the pipeline: `python scripts/run_pipeline.py`

Baseline pipeline
-----------------
- The default config runs the **baseline** edit loop (plan → edit → critic) using
  `black-forest-labs/flux-kontext-max`. The critic compares the original vs edited
  image and only accepts edits that are both realistic and minimal.
- To switch baseline models, edit `baseline_model` in `src/config.py`.
- The baseline walkthrough lives in `notebooks/pipeline_walkthrough.ipynb`.

Notes
-----
- Segmentation is implemented as a **Grounded-SAM-2 style** pipeline: Grounding DINO (text → box) followed by SAM 2 (box → mask). The reference we follow is [IDEA-Research/Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2/tree/main).
- The current implementation scaffolds the LangGraph workflow and API wrappers with placeholders for model IDs and prompts. Swap in your Replicate model hashes and refine prompts as you iterate. Public slugs observed on Replicate’s catalog you can try (verify access in your account): `adirik/grounding-dino:latest` for detection, `meta/sam-2:latest` for masks, `black-forest-labs/flux-fill-pro:latest` for inpainting, and the baseline editor `black-forest-labs/flux-kontext-max`. Update `src/config.py` if different in your workspace. See model pages for details: [Grounding DINO](https://replicate.com/adirik/grounding-dino), [FLUX Fill Pro](https://replicate.com/black-forest-labs/flux-fill-pro).

