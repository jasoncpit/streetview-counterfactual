from __future__ import annotations 

import logging
import os
import re
import shutil
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
# Compatibility models
# ---------------------------------------------------------------------------


@dataclass
class ReplicateModels:
    dino_model: str = ""
    sam_model: str = ""
    inpaint_model: str = ""
    flux_kontext_model: str = "black-forest-labs/flux-kontext-pro"
    nano_banana_model: str = "google/nano-banana-pro"
    inpaint_params: Dict[str, Any] | None = None
    flux_kontext_params: Dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class ReplicateClient:
    """Thin wrapper around the Replicate API for segmentation, inpainting,
    baseline image-editing, and realism scoring."""

    def __init__(self, download_timeout: int = 120, models: ReplicateModels | None = None) -> None:
        self.client = replicate.Client(api_token=os.getenv("REPLICATE_API_TOKEN"))
        self.download_timeout = download_timeout
        self.models = models
        self._next_prediction_create_at = 0.0

    def image_edit_baseline(
        self,
        model: Literal[
            "google/nano-banana-pro",
            "bytedance/seedream-4",
            "openai/gpt-image-1.5",
            "black-forest-labs/flux-kontext-pro",
            "qwen/qwen-image-edit",
            "qwen/qwen-image-edit-plus"
        ],
        image_path: str,
        edit_plan: str,
        target_object: str,
        output_dir: Path,
        prompt_template: str,
        lever_concept: str = "",
        lever_family: str = "",
        scene_support: str = "",
        intervention_direction: str = "",
        edit_template: str = "",
        api_max_retries: int = 3,
        api_retry_base_delay: float = 1.0,
        prediction_timeout_s: float = 90.0,
        match_input_size: bool = True,
    ) -> Path:
        prompt = prompt_template.format(
            lever_concept=lever_concept,
            lever_family=lever_family,
            scene_support=scene_support,
            intervention_direction=intervention_direction,
            edit_template=edit_template,
            target_object=target_object,
            edit_plan=edit_plan,
        )
        self.last_baseline_used_mock = False

        result, output_format = self._run_with_transport_retries(
            model,
            image_path,
            prompt,
            api_max_retries,
            api_retry_base_delay,
            prediction_timeout_s,
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

    def _run_with_transport_retries(
        self,
        model: str,
        image_path: str,
        prompt: str,
        api_max_retries: int,
        api_retry_base_delay: float,
        prediction_timeout_s: float,
    ) -> tuple[Any, str | None]:
        """Run a baseline edit with exponential-backoff transport retries.

        Returns ``(result, output_format)`` on success, or ``(None, None)``
        when all attempts are exhausted.
        """
        for attempt in range(1, api_max_retries + 1):
            try:
                with open(image_path, "rb") as fh:
                    payload, fmt = self._build_baseline_payload(
                        model, fh, prompt,
                        use_alt=(model == "openai/gpt-image-1.5" and attempt > 1),
                    )
                    logger.info(
                        "Running %s (transport attempt %s/%s)",
                        model,
                        attempt,
                        api_max_retries,
                    )
                    result = self._run_prediction_with_timeout(
                        model=model,
                        payload=payload,
                        timeout_s=prediction_timeout_s,
                    )
                return self._normalize_result(result), fmt
            except (ReplicateError, ModelError, TimeoutError, RuntimeError) as err:
                if attempt >= api_max_retries:
                    logger.warning(
                        "Baseline edit failed after %s transport retries: %s",
                        api_max_retries,
                        err,
                        exc_info=True,
                    )
                    return None, None
                delay = self._retry_delay_seconds(
                    err,
                    attempt=attempt,
                    api_retry_base_delay=api_retry_base_delay,
                )
                self._next_prediction_create_at = max(
                    self._next_prediction_create_at,
                    time.monotonic() + delay,
                )
                logger.warning(
                    "Transport attempt %s/%s failed: %s; retrying in %.1fs.",
                    attempt,
                    api_max_retries,
                    err,
                    delay,
                )
                time.sleep(delay)
        return None, None  # unreachable, keeps type-checkers happy

    def _run_prediction_with_timeout(
        self,
        *,
        model: str,
        payload: Dict[str, Any],
        timeout_s: float,
    ) -> Any:
        wait_s = self._next_prediction_create_at - time.monotonic()
        if wait_s > 0:
            logger.info("Waiting %.1fs for Replicate prediction create window.", wait_s)
            time.sleep(wait_s)

        owner, name = model.split("/", 1)
        prediction = self.client.models.predictions.create(
            model=(owner, name),
            input=payload,
            wait=False,
        )
        deadline = time.monotonic() + timeout_s
        while prediction.status not in {"succeeded", "failed", "canceled"}:
            if time.monotonic() >= deadline:
                try:
                    prediction.cancel()
                except Exception:
                    logger.warning("Failed to cancel timed-out prediction %s", prediction.id)
                raise TimeoutError(
                    f"Replicate prediction timed out after {timeout_s:.1f}s: {prediction.id}"
                )
            time.sleep(self.client.poll_interval)
            prediction.reload()

        if prediction.status == "succeeded":
            return prediction.output

        raise RuntimeError(
            f"Replicate prediction {prediction.id} ended with status={prediction.status}: "
            f"{prediction.error or 'no error message'}"
        )

    def _retry_delay_seconds(
        self,
        err: Exception,
        *,
        attempt: int,
        api_retry_base_delay: float,
    ) -> float:
        delay = api_retry_base_delay * (2 ** (attempt - 1))
        message = str(err)
        if "status: 429" not in message and "Request was throttled" not in message:
            return delay

        match = re.search(r"resets in ~(\d+(?:\.\d+)?)s", message)
        if match:
            return max(delay, float(match.group(1)) + 1.0)
        return max(delay, 10.0)

    # ── baseline: payload builders ────────────────────────────────────────

    def _build_baseline_payload(
        self,
        model: str,
        image_handle: Any,
        prompt: str | None = None,
        *,
        formatted_prompt: str | None = None,
        use_alt: bool = False,
    ) -> tuple[Dict[str, Any], str | None]:
        """Return ``(payload, output_format)`` for the given baseline model."""
        prompt = formatted_prompt or prompt or ""
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

        if model in {
            "black-forest-labs/flux-kontext-max",
            "black-forest-labs/flux-kontext-pro",
        }:
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

    def _normalize_result(self, result: Any) -> Any:
        return self._first_item(result)

    @staticmethod
    def _suffix_from_format(fmt: str | None) -> str:
        if not fmt:
            return ".png"
        return ".jpg" if str(fmt).lower() in {"jpg", "jpeg"} else ".png"

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

    def _mock_inpaint(self, image_path: str, output_dir: Path) -> Path:
        """Fallback output used when remote generation fails entirely.

        This preserves the input image dimensions and allows the caller to
        record an explicit invalid/mocked attempt rather than crashing the
        whole image-level pipeline.
        """
        source = Path(image_path)
        suffix = source.suffix or ".png"
        destination = timestamped_path(output_dir, "image_edit_baseline_mock", suffix=suffix)
        ensure_dir(destination.parent)
        shutil.copy2(source, destination)
        return destination

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

    def _match_size(self, output_path: Path, input_path: str) -> None:
        self._match_size_safe(output_path, input_path)
