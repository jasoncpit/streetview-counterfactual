from __future__ import annotations 

import logging
import os
import tempfile
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import replicate
from dotenv import load_dotenv
from PIL import Image
from replicate.exceptions import ModelError, ReplicateError
from typing_extensions import Literal

from src.utils.paths import ensure_dir, timestamped_path

load_dotenv()

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class ReplicateClient:
    """Thin wrapper around the Replicate API for segmentation, inpainting,
    baseline image-editing, and realism scoring."""

    def __init__(self, download_timeout: int = 120) -> None:
        self.client = replicate.Client(api_token=os.getenv("REPLICATE_API_TOKEN"))
        self.download_timeout = download_timeout

    def image_edit_baseline(
        self,
        model: Literal[
            "google/nano-banana-pro",
            "bytedance/seedream-4",
            "openai/gpt-image-1.5",
            "black-forest-labs/flux-kontext-max",
            "qwen/qwen-image-edit",
        ],
        image_path: str,
        edit_plan: str,
        target_object: str,
        output_dir: Path,
        prompt_template: str,
        max_retries: int = 3,
        retry_base_delay: float = 1.0,
        match_input_size: bool = True,
    ) -> Path:
        prompt = prompt_template.format(
            target_object=target_object, edit_plan=edit_plan,
        )
        self.last_baseline_used_mock = False

        result, output_format = self._run_with_retries(
            model, image_path, prompt, max_retries, retry_base_delay,
        )

        if result is None:
            self.last_baseline_used_mock = True
            return self._mock_inpaint(image_path, output_dir)

        suffix = ".jpg" if str(output_format).lower() in {"jpg", "jpeg"} else ".png"
        output_path = timestamped_path(output_dir, "image_edit_baseline", suffix=suffix)
        if not self._save_result(result, output_path):
            logger.warning("Unrecognised result type (%s); falling back to mock.", type(result))
            self.last_baseline_used_mock = True
            return None

        if match_input_size:
            self._match_size_safe(output_path, image_path)
        return output_path
        
    def segment_object(
        self,
        image_path: str,
        prompt: str,
        mask_dir: Path,
        *,
        box_threshold: float | None = None,
        text_threshold: float | None = None,
    ) -> Path:
        pass 

    def inpaint(self, image_path: str, mask_path: str, prompt: str, output_dir: Path) -> Path:
        pass 


    # ── baseline: retry loop ──────────────────────────────────────────────

    def _run_with_retries(
        self,
        model: str,
        image_path: str,
        prompt: str,
        max_retries: int,
        retry_base_delay: float,
    ) -> tuple[Any, str | None]:
        """Run a baseline edit with exponential-backoff retries.

        Returns ``(result, output_format)`` on success, or ``(None, None)``
        when all attempts are exhausted.
        """
        for attempt in range(1, max_retries + 1):
            try:
                with open(image_path, "rb") as fh:
                    payload, fmt = self._build_baseline_payload(
                        model, fh, prompt,
                        use_alt=(model == "openai/gpt-image-1.5" and attempt > 1),
                    )
                    logger.info("Running %s (attempt %s/%s)", model, attempt, max_retries)
                    result = self.client.run(model, input=payload)
                return self._first_item(result), fmt
            except (ReplicateError, ModelError) as err:
                if attempt >= max_retries:
                    logger.warning(
                        "Baseline edit failed after %s attempts: %s",
                        max_retries, err, exc_info=True,
                    )
                    return None, None
                delay = retry_base_delay * (2 ** (attempt - 1))
                logger.warning(
                    "Attempt %s/%s failed: %s — retrying in %.1fs.",
                    attempt, max_retries, err, delay,
                )
                time.sleep(delay)
        return None, None  # unreachable, keeps type-checkers happy

    # ── baseline: payload builders ────────────────────────────────────────

    def _build_baseline_payload(
        self, model: str, image_handle: Any, prompt: str, *, use_alt: bool = False,
    ) -> tuple[Dict[str, Any], str | None]:
        """Return ``(payload, output_format)`` for the given baseline model."""
        if model == "google/nano-banana-pro":
            return {
                "prompt": prompt,
                "resolution": "2K",
                "image_input": [image_handle],
                "aspect_ratio": "match_input_image",
                "output_format": "png",
                "safety_filter_level": "block_only_high",
            }, "png"

        if model == "bytedance/seedream-4":
            return {
                "size": "2K",
                "width": 2048,
                "height": 2048,
                "prompt": prompt,
                "max_images": 1,
                "image_input": [image_handle],
                "aspect_ratio": "match_input_image",
                "enhance_prompt": True,
                "sequential_image_generation": "disabled",
            }, None

        if model == "openai/gpt-image-1.5":
            image_key = "image" if use_alt else "input_images"
            return {
                "prompt": prompt,
                "quality": "high",
                "background": "auto",
                "moderation": "auto",
                "aspect_ratio": "match_input_image",
                "output_format": "png",
                "input_fidelity": "high",
                "number_of_images": 1,
                "output_compression": 90,
                image_key: [image_handle],
            }, "png"

        if model == "black-forest-labs/flux-kontext-max":
            return {
                "prompt": prompt,
                "input_image": image_handle,
                "aspect_ratio": "match_input_image",
                "output_format": "jpg",
                "safety_tolerance": 2,
                "prompt_upsampling": False,
            }, "jpg"        

        if model == "qwen/qwen-image-edit":
            return {
                "prompt": prompt,
                "image": image_handle,
                "go_fast": True,
                "aspect_ratio": "match_input_image",
                "output_format": "jpg",
                "output_compression": 90,
            }, "jpg"

        raise ValueError(f"Unsupported baseline model: {model}")

    # ── segmentation internals ────────────────────────────────────────────

    def _segment_dino_then_sam(
        self,
        image_path: str,
        prompt: str,
        mask_dir: Path,
        *,
        box_threshold: float | None = None,
        text_threshold: float | None = None,
    ) -> Path:
        pass 

    def _segment_grounded_sam(self, image_path: str, prompt: str, mask_dir: Path) -> Path:
        """Single-model segmentation using a bundled GroundingDINO+SAM model."""
        pass 

    # ── output helpers ────────────────────────────────────────────────────

    @staticmethod
    def _first_item(result: Any) -> Any:
        """Unwrap single-element list/tuple returned by some Replicate models."""
        if isinstance(result, (list, tuple)) and result:
            return result[0]
        return result

    def _save_result(self, result: Any, destination: Path) -> bool:
        """Persist a Replicate result to *destination*.

        Handles ``FileOutput``, plain URL strings, and generic file-like
        objects.  Returns ``True`` on success, ``False`` if the result type
        is unrecognised.
        """
        ensure_dir(destination.parent)
        # FileOutput or any file-like object
        if isinstance(result, replicate.helpers.FileOutput) or hasattr(result, "read"):
            destination.write_bytes(result.read())
            return True
        # Plain URL string
        if isinstance(result, str):
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                urllib.request.urlretrieve(result, tmp.name)
                Path(tmp.name).replace(destination)
            return True
        return False

    # ── bounding-box helpers ──────────────────────────────────────────────

    def _normalize_boxes(self, bbox_raw: Any) -> List[List[float]]:
        """Coerce varied GroundingDINO outputs into ``[[x1, y1, x2, y2], …]``."""
        pass 



    # ── image resize ──────────────────────────────────────────────────────

    def _match_size_safe(self, output_path: Path, input_path: str) -> None:
        """Resize *output_path* to match the dimensions of *input_path*."""
        try:
            input_img = Image.open(input_path)
            output_img = Image.open(output_path)
            if input_img.size == output_img.size:
                return
            in_w, in_h = input_img.size
            out_w, out_h = output_img.size
            scale = max(in_w / out_w, in_h / out_h)
            new_w = max(1, int(round(out_w * scale)))
            new_h = max(1, int(round(out_h * scale)))
            resized = output_img.resize((new_w, new_h), resample=Image.LANCZOS)
            left = (new_w - in_w) // 2
            top = (new_h - in_h) // 2
            resized.crop((left, top, left + in_w, top + in_h)).save(output_path)
        except Exception:
            logger.warning("Failed to resize baseline output to match input.", exc_info=True)
