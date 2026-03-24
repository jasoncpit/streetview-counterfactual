# Remote Repo Inspection

Source inspected without cloning: <https://github.com/strawmelon11/human-perception-place-pulse>

## Observed file tree

- `README.md`
- `requirements.txt`
- `Model_01.py`
- `eval.py`
- `test_image/` with three sample images
- `LICENSE`

## Repo behavior

- The project is inference-only as published.
- `Model_01.py` defines a ViT-B/16 backbone using `torchvision.models.vit_b_16` with a 3-layer MLP classification head.
- `eval.py` downloads six pretrained `.pth` checkpoints from Hugging Face repo `Jiani11/human-perception-place-pulse`.
- The six output attributes are `safety`, `lively`, `wealthy`, `beautiful`, `boring`, and `depressing`.
- Inference resizes images to `384x384`, normalizes with ImageNet mean/std, applies a 2-class softmax head, and scales the positive-class probability to a `0-10` score.
- The script expects a flat input image directory and writes one CSV per perception attribute.

## Dependency surface

`requirements.txt` lists:

- `pandas`
- `Pillow`
- `torch`
- `torchvision`
- `huggingface_hub`
- `transformers`

## Implication for pretraining

The remote repo does not include training code or a dataset formatter. The local workspace here fills that gap by downloading SPECS and producing normalized manifests that are easier to use for classifier pretraining or fine-tuning.
