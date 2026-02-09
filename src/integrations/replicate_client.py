from __future__ import annotations

import logging
import os
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence
from typing_extensions import Literal

import replicate
from PIL import Image
from dotenv import load_dotenv
from replicate.exceptions import ReplicateError, ModelError

from src.utils.paths import ensure_dir, timestamped_path

load_dotenv()

logger = logging.getLogger(__name__)


@dataclass
class ReplicateModels:
    dino_model: str
    sam_model: str
    inpaint_model: str
    flux_kontext_model: str
    nano_banana_model: str
    mock: bool = False
    inpaint_params: Dict[str, Any] | None = None
    flux_kontext_params: Dict[str, Any] | None = None
    dino_params: Dict[str, Any] | None = None
    sam_params: Dict[str, Any] | None = None

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "ReplicateModels":
        return cls(
            dino_model=cfg["dino_model"],
            sam_model=cfg.get("sam_model") or cfg.get("sam3_model"),
            inpaint_model=cfg["inpaint_model"],
            flux_kontext_model=cfg.get("flux_kontext_model", "black-forest-labs/flux-kontext-max"),
            nano_banana_model=cfg["nano_banana_model"],
            mock=bool(cfg.get("mock", False)),
            inpaint_params=cfg.get("inpaint_params"),
            flux_kontext_params=cfg.get("flux_kontext_params"),
            dino_params=cfg.get("dino_params"),
            sam_params=cfg.get("sam_params"),
        )


class ReplicateClient:
    def __init__(self, models: ReplicateModels, download_timeout: int = 120) -> None:
        self.client = replicate.Client(api_token=os.getenv("REPLICATE_API_TOKEN"))
        self.models = models
        self.download_timeout = download_timeout
        self.last_baseline_used_mock: bool = False

    def image_edit_baseline(
        self,
        model: Literal[
            "google/nano-banana-pro",
            "bytedance/seedream-4",
            "openai/gpt-image-1.5",
            "black-forest-labs/flux-kontext-max",
        ],
        image_path: str,
        edit_plan: str,
        target_object: str,
        output_dir: Path,
        *,
        match_input_size: bool = True,
        max_retries: int = 3,
        retry_base_delay: float = 1.0,
    ) -> Path:
        base_prompt = (
            "Use the provided image as the base. Do NOT generate a new scene. "
            "Preserve the exact camera viewpoint, geometry, lighting, color, and all objects. "
            "Only edit the target object and only as much as required by the plan. "
            "Follow the edit plan literally; do not embellish or add extra elements. "
            "Do not add or remove any other objects, signs, markings, people, vehicles, or text. "
            "If the plan mentions repainting existing markings, keep their layout, spacing, and color; "
            "only adjust brightness/texture/width subtly. "
            "If the plan mentions adding a marking, add only that marking and nothing else. "
            "Keep everything else pixel-identical outside the target area. "
            "Object: {target_object}. Edit plan: {edit_plan}"
        )
        formatted_prompt = base_prompt.format(target_object=target_object, edit_plan=edit_plan)
        self.last_baseline_used_mock = False

        result = None
        output_format = None
        for attempt in range(1, max_retries + 1):
            try:
                with open(image_path, "rb") as image_handle:
                    payload, output_format = self._build_baseline_payload(
                        model,
                        image_handle,
                        formatted_prompt,
                        use_alt=(model == "openai/gpt-image-1.5" and attempt > 1),
                    )
                    logger.info("Running model: %s with payload: %s", model, payload)
                    result = self.client.run(model, input=payload)
                logger.info("Result URL: %s", result)
                break
            except (ReplicateError, ModelError) as err:
                if attempt >= max_retries:
                    logger.warning(
                        "Baseline edit failed after %s attempts; falling back to mock image. model=%s",
                        max_retries,
                        model,
                        exc_info=True,
                    )
                    self.last_baseline_used_mock = True
                    return self._mock_inpaint(image_path, output_dir)
                delay = retry_base_delay * (2 ** (attempt - 1))
                logger.warning(
                    "Baseline edit failed (attempt %s/%s): %s. Retrying in %.2fs.",
                    attempt,
                    max_retries,
                    err,
                    delay,
                )
                import time

                time.sleep(delay)

        result = self._normalize_result(result)
        suffix = self._suffix_from_format(output_format)
        output_path = timestamped_path(output_dir, "image_edit_baseline", suffix=suffix)
        if not self._write_output(result, output_path):
            logger.warning("Invalid result URL: %s; falling back to mock image.", result)
            self.last_baseline_used_mock = True
            return self._mock_inpaint(image_path, output_dir)

        if match_input_size:
            try:
                self._match_size(output_path, image_path)
            except Exception:
                logger.warning("Failed to resize baseline output to match input.", exc_info=True)
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
        if self.models.mock:
            logger.info("Segment (mock) for %s with prompt='%s'", image_path, prompt)
            return self._mock_mask(image_path, mask_dir)
        try:
            if "grounded_sam" in (self.models.sam_model or ""):
                return self._segment_grounded_sam(image_path, prompt, mask_dir)

            logger.info(
                "Segment start | dino_model=%s | sam_model=%s | image=%s | prompt=%s",
                self.models.dino_model,
                self.models.sam_model,
                image_path,
                prompt,
            )
            with open(image_path, "rb") as image_handle:
                dino_input: Dict[str, Any] = {"image": image_handle, "prompt": prompt}
                if box_threshold is not None:
                    dino_input["box_threshold"] = box_threshold
                if text_threshold is not None:
                    dino_input["text_threshold"] = text_threshold
                if self.models.dino_params:
                    dino_input.update(self.models.dino_params)
                bbox_raw = self.client.run(self.models.dino_model, input=dino_input)

            logger.debug("DINO bbox result: %s", bbox_raw)
            boxes = self._normalize_boxes(bbox_raw)
            if not boxes:
                raise ReplicateError("GroundingDINO returned no boxes")

            box_for_sam = self._merge_boxes(boxes)
            with open(image_path, "rb") as image_handle:
                sam_input: Dict[str, Any] = {"image": image_handle, "box": box_for_sam}
                if self.models.sam_params:
                    sam_input.update(self.models.sam_params)
                mask_url = self.client.run(self.models.sam_model, input=sam_input)

            logger.debug("SAM mask url: %s", mask_url)
            ensure_dir(mask_dir)
            output_path = timestamped_path(mask_dir, Path(image_path).stem, suffix=".png")
            self._download_url(mask_url, output_path)
            logger.info("Segment success -> %s", output_path)
            return output_path
        except (ReplicateError, ModelError) as err:
            logger.warning(
                "Segment failed via Replicate (%s); falling back to mock mask. models=[%s,%s]",
                err,
                self.models.dino_model,
                self.models.sam_model,
                exc_info=True,
            )
            return self._mock_mask(image_path, mask_dir)

    def inpaint(self, image_path: str, mask_path: str, prompt: str, output_dir: Path) -> Path:
        if self.models.mock:
            logger.info("Inpaint (mock) for %s using mask %s", image_path, mask_path)
            return self._mock_inpaint(image_path, output_dir)
        try:
            logger.info(
                "Inpaint start | model=%s | image=%s | mask=%s | prompt=%s",
                self.models.inpaint_model,
                image_path,
                mask_path,
                prompt,
            )
            ensure_dir(output_dir)
            with open(image_path, "rb") as image_handle, open(mask_path, "rb") as mask_handle:
                payload = {
                    "image": image_handle,
                    "mask": mask_handle,
                    "prompt": prompt,
                }
                if self.models.inpaint_params:
                    payload.update(self.models.inpaint_params)
                result = self.client.run(self.models.inpaint_model, input=payload)

            result = self._normalize_result(result)
            output_path = timestamped_path(output_dir, Path(image_path).stem, suffix=".png")
            if not self._write_output(result, output_path):
                raise ReplicateError("Inpaint returned unsupported output type")
            logger.info("Inpaint success -> %s", output_path)
            return output_path
        except (ReplicateError, ModelError) as err:
            logger.warning(
                "Inpaint failed via Replicate (%s); falling back to mock image copy. model=%s",
                err,
                self.models.inpaint_model,
                exc_info=True,
            )
            return self._mock_inpaint(image_path, output_dir)

    def score_realism(self, image_path: str) -> float:
        if self.models.mock:
            logger.info("Realism (mock) for %s -> 0.5", image_path)
            return 0.5
        try:
            logger.info("Realism start | model=%s | image=%s", self.models.nano_banana_model, image_path)
            with open(image_path, "rb") as image_handle:
                score = self.client.run(
                    self.models.nano_banana_model,
                    input={"image": image_handle},
                )
            numeric = float(score) if score is not None else 0.0
            logger.info("Realism success -> %.3f", numeric)
            return numeric
        except (ReplicateError, ModelError) as err:
            logger.warning(
                "Realism scoring failed via Replicate (%s); falling back to 0.5. model=%s",
                err,
                self.models.nano_banana_model,
                exc_info=True,
            )
            return 0.5

    def _build_baseline_payload(
        self,
        model: str,
        image_handle,
        formatted_prompt: str,
        *,
        use_alt: bool = False,
    ) -> tuple[Dict[str, Any], str | None]:
        if model == "google/nano-banana-pro":
            payload = {
                "prompt": formatted_prompt,
                "resolution": "2K",
                "image_input": [image_handle],
                "aspect_ratio": "match_input_image",
                "output_format": "png",
                "safety_filter_level": "block_only_high",
            }
            return payload, payload.get("output_format")
        if model == "bytedance/seedream-4":
            payload = {
                "size": "2K",
                "width": 2048,
                "height": 2048,
                "prompt": formatted_prompt,
                "max_images": 1,
                "image_input": [image_handle],
                "aspect_ratio": "match_input_image",
                "enhance_prompt": True,
                "sequential_image_generation": "disabled",
            }
            return payload, payload.get("output_format")
        if model == "openai/gpt-image-1.5":
            image_key = "image" if use_alt else "input_images"
            payload = {
                "prompt": formatted_prompt,
                "quality": "high",
                "background": "auto",
                "moderation": "auto",
                "aspect_ratio": "match_input_image",
                "output_format": "png",
                "input_fidelity": "high",
                "number_of_images": 1,
                "output_compression": 90,
                image_key: [image_handle],
            }
            return payload, payload.get("output_format")
        if model == self.models.flux_kontext_model or model == "black-forest-labs/flux-kontext-max":
            payload = {
                "prompt": formatted_prompt,
                "input_image": image_handle,
            }
            if self.models.flux_kontext_params:
                payload.update(self.models.flux_kontext_params)
            return payload, payload.get("output_format")
        raise ValueError(f"Invalid model: {model}")

    def _normalize_result(self, result: Any) -> Any:
        if isinstance(result, (list, tuple)) and result:
            return result[0]
        return result

    def _write_output(self, result: Any, output_path: Path) -> bool:
        if isinstance(result, replicate.helpers.FileOutput):
            with open(output_path, "wb") as f:
                f.write(result.read())
            return True
        if isinstance(result, str):
            self._download_url(result, output_path)
            return True
        if hasattr(result, "read"):
            with open(output_path, "wb") as f:
                f.write(result.read())
            return True
        return False

    def _download_url(self, url: str, destination: Path) -> None:
        if isinstance(url, replicate.helpers.FileOutput):
            with open(destination, "wb") as f:
                f.write(url.read())
            return
        logger.debug("Downloading %s -> %s", url, destination)
        with tempfile.NamedTemporaryFile(delete=False) as handle:
            urllib.request.urlretrieve(url, handle.name)
            ensure_dir(destination.parent)
            Path(handle.name).replace(destination)

    def _segment_grounded_sam(self, image_path: str, prompt: str, mask_dir: Path) -> Path:
        logger.info(
            "Segment start | grounded_sam_model=%s | image=%s | prompt=%s",
            self.models.sam_model,
            image_path,
            prompt,
        )
        with open(image_path, "rb") as image_handle:
            result = self.client.run(
                self.models.sam_model,
                input={
                    "image": image_handle,
                    "mask_prompt": prompt,
                    "negative_mask_prompt": "",
                    "adjustment_factor": 0,
                },
            )
        mask_url = None
        if isinstance(result, (list, tuple)) and result:
            mask_url = result[0]
        elif isinstance(result, str):
            mask_url = result
        logger.debug("Grounded SAM mask url: %s", mask_url)
        if not mask_url:
            raise ReplicateError("Grounded SAM returned empty result")

        ensure_dir(mask_dir)
        output_path = timestamped_path(mask_dir, Path(image_path).stem, suffix=".png")
        self._download_url(mask_url, output_path)
        logger.info("Segment success (grounded_sam) -> %s", output_path)
        return output_path

    def _normalize_boxes(self, bbox_raw: Any) -> List[List[float]]:
        if bbox_raw is None:
            return []
        if isinstance(bbox_raw, (list, tuple)):
            if len(bbox_raw) == 4 and all(isinstance(v, (int, float)) for v in bbox_raw):
                return [[float(v) for v in bbox_raw]]
            boxes: List[List[float]] = []
            for item in bbox_raw:
                if isinstance(item, (list, tuple)) and len(item) == 4:
                    try:
                        boxes.append([float(v) for v in item])
                    except (TypeError, ValueError):
                        continue
            return boxes
        if isinstance(bbox_raw, dict):
            inner = bbox_raw.get("boxes") or bbox_raw.get("box") or bbox_raw.get("bboxes")
            if inner is not None:
                return self._normalize_boxes(inner)
        if isinstance(bbox_raw, str):
            import json

            try:
                parsed = json.loads(bbox_raw)
            except Exception:
                return []
            return self._normalize_boxes(parsed)
        return []

    def _merge_boxes(self, boxes: Sequence[Sequence[float]]) -> List[float]:
        xs1 = [b[0] for b in boxes]
        ys1 = [b[1] for b in boxes]
        xs2 = [b[2] for b in boxes]
        ys2 = [b[3] for b in boxes]
        return [float(min(xs1)), float(min(ys1)), float(max(xs2)), float(max(ys2))]

    def _suffix_from_format(self, output_format: str | None) -> str:
        if output_format and str(output_format).lower() in {"jpg", "jpeg"}:
            return ".jpg"
        return ".png"

    def _mock_mask(self, image_path: str, mask_dir: Path) -> Path:
        ensure_dir(mask_dir)
        img = Image.open(image_path)
        mask = Image.new("L", img.size, color=255)
        output_path = timestamped_path(mask_dir, Path(image_path).stem, suffix=".png")
        mask.save(output_path)
        return output_path

    def _mock_inpaint(self, image_path: str, output_dir: Path) -> Path:
        ensure_dir(output_dir)
        img = Image.open(image_path)
        output_path = timestamped_path(output_dir, Path(image_path).stem, suffix=".png")
        img.save(output_path)
        return output_path

    def _match_size(self, output_path: Path, input_path: str) -> None:
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
        left = max(0, (new_w - in_w) // 2)
        top = max(0, (new_h - in_h) // 2)
        cropped = resized.crop((left, top, left + in_w, top + in_h))
        cropped.save(output_path)
