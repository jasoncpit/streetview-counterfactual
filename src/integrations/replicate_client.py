import logging
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import replicate
from PIL import Image
from replicate.exceptions import ReplicateError

from src.utils.paths import ensure_dir, timestamped_path

logger = logging.getLogger(__name__)


@dataclass
class ReplicateModels:
    dino_model: str
    sam_model: str
    inpaint_model: str
    nano_banana_model: str
    mock: bool = False
    inpaint_params: Dict[str, Any] | None = None
    dino_params: Dict[str, Any] | None = None
    sam_params: Dict[str, Any] | None = None

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "ReplicateModels":
        return cls(
            dino_model=cfg["dino_model"],
            sam_model=cfg.get("sam_model") or cfg.get("sam3_model"),
            inpaint_model=cfg["inpaint_model"],
            nano_banana_model=cfg["nano_banana_model"],
            mock=bool(cfg.get("mock", False)),
            inpaint_params=cfg.get("inpaint_params"),
            dino_params=cfg.get("dino_params"),
            sam_params=cfg.get("sam_params"),
        )


class ReplicateClient:
    def __init__(self, models: ReplicateModels, download_timeout: int = 120) -> None:
        self.client = replicate.Client()
        self.models = models
        self.download_timeout = download_timeout

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
            dino_input: Dict[str, Any] = {"image": open(image_path, "rb"), "prompt": prompt}
            # Grounded-SAM-2 style knobs (only included if configured).
            # Some Replicate DINO wrappers accept these args; others will ignore/reject them.
            # We keep them optional to preserve backwards compatibility.
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
            sam_input: Dict[str, Any] = {"image": open(image_path, "rb"), "box": box_for_sam}
            if self.models.sam_params:
                sam_input.update(self.models.sam_params)

            mask_url = self.client.run(self.models.sam_model, input=sam_input)
            logger.debug("SAM mask url: %s", mask_url)
            ensure_dir(mask_dir)
            output_path = timestamped_path(mask_dir, Path(image_path).stem, suffix=".png")
            self._download(mask_url, output_path)
            logger.info("Segment success -> %s", output_path)
            return output_path
        except ReplicateError as err:
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
            payload = {
                "image": open(image_path, "rb"),
                "mask": open(mask_path, "rb"),
                "prompt": prompt,
            }
            if self.models.inpaint_params:
                payload.update(self.models.inpaint_params)

            result_url = self.client.run(
                self.models.inpaint_model,
                input=payload,
            )
            logger.debug("Inpaint result url: %s", result_url)
            output_path = timestamped_path(output_dir, Path(image_path).stem, suffix=".png")
            self._download(result_url, output_path)
            logger.info("Inpaint success -> %s", output_path)
            return output_path
        except ReplicateError as err:
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
            score = self.client.run(
                self.models.nano_banana_model,
                input={"image": open(image_path, "rb")},
            )
            numeric = float(score) if score is not None else 0.0
            logger.info("Realism success -> %.3f", numeric)
            return numeric
        except ReplicateError as err:
            logger.warning(
                "Realism scoring failed via Replicate (%s); falling back to 0.5. model=%s",
                err,
                self.models.nano_banana_model,
                exc_info=True,
            )
            return 0.5

    def _download(self, url: str, destination: Path) -> None:
        logger.debug("Downloading %s -> %s", url, destination)
        with tempfile.NamedTemporaryFile(delete=False) as handle:
            urllib.request.urlretrieve(url, handle.name)
            ensure_dir(destination.parent)
            Path(handle.name).replace(destination)

    def _segment_grounded_sam(self, image_path: str, prompt: str, mask_dir: Path) -> Path:
        """Use schananas/grounded_sam which bundles GroundingDINO+SAM."""
        logger.info(
            "Segment start | grounded_sam_model=%s | image=%s | prompt=%s",
            self.models.sam_model,
            image_path,
            prompt,
        )
        result = self.client.run(
            self.models.sam_model,
            input={
                "image": open(image_path, "rb"),
                "mask_prompt": prompt,
                "negative_mask_prompt": "",
                "adjustment_factor": 0,
            },
        )
        # Model can stream list of outputs; pick first url-like item.
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
        self._download(mask_url, output_path)
        logger.info("Segment success (grounded_sam) -> %s", output_path)
        return output_path

    def _normalize_boxes(self, bbox_raw: Any) -> List[List[float]]:
        """
        Normalize GroundingDINO output into a list of [x1, y1, x2, y2] boxes.

        Replicate wrappers vary: some return a single box, some return a list of boxes,
        some return dicts like {"boxes": [...]}.
        """
        if bbox_raw is None:
            return []
        # Common: single box [x1,y1,x2,y2]
        if isinstance(bbox_raw, (list, tuple)):
            # List of 4 numbers
            if len(bbox_raw) == 4 and all(isinstance(v, (int, float)) for v in bbox_raw):
                return [[float(v) for v in bbox_raw]]
            # List of boxes
            boxes: List[List[float]] = []
            for item in bbox_raw:
                if isinstance(item, (list, tuple)) and len(item) == 4:
                    try:
                        boxes.append([float(v) for v in item])
                    except (TypeError, ValueError):
                        continue
            return boxes
        # Dict form: {"boxes": [...]} (optionally with scores/labels)
        if isinstance(bbox_raw, dict):
            inner = bbox_raw.get("boxes") or bbox_raw.get("box") or bbox_raw.get("bboxes")
            if inner is not None:
                return self._normalize_boxes(inner)
        # Stringified JSON-ish output (best effort)
        if isinstance(bbox_raw, str):
            import json

            try:
                parsed = json.loads(bbox_raw)
            except Exception:
                return []
            return self._normalize_boxes(parsed)
        return []

    def _merge_boxes(self, boxes: Sequence[Sequence[float]]) -> List[float]:
        """Merge many boxes into a single bounding box that covers them all."""
        xs1 = [b[0] for b in boxes]
        ys1 = [b[1] for b in boxes]
        xs2 = [b[2] for b in boxes]
        ys2 = [b[3] for b in boxes]
        return [float(min(xs1)), float(min(ys1)), float(max(xs2)), float(max(ys2))]

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

