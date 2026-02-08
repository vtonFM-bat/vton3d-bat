# vton3d/utils/background_segmentation.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
from PIL import Image

import torch
from transformers import Sam3Processor, Sam3Model


@dataclass
class BackgroundSegmentationConfig:
    model_id: str = "facebook/sam3"
    prompt: str = "human"

    threshold: float = 0.5
    mask_threshold: float = 0.5

    pick: str = "union"  # "union" | "largest" | "best_score"
    overwrite: bool = True

    device: Optional[str] = None

    # Saving masks
    masks_dir_name: str = "human_masks"  # created under <scene_dir>/
    mask_suffix: str = ""               # optional suffix like "_human"

    # W&B logging
    wandb_log: bool = True
    wandb_prefix: str = "bgseg"         # e.g. bgseg/<name>
    overlay_alpha: float = 0.45
    overlay_color_rgb: Tuple[int, int, int] = (255, 100, 180)  # just for debug overlay


class BackgroundSegmentation:
    """
    - segments human with SAM3 text prompt
    - writes mask as PNG into <scene_dir>/human_masks/<name>.png
    - whites background in-place for qwen/images/<name>.png
    - logs to wandb: segmented image, mask, overlay
    """

    def __init__(self, cfg: BackgroundSegmentationConfig):
        self.cfg = cfg

        if cfg.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = cfg.device

        if cfg.pick not in ("union", "largest", "best_score"):
            raise ValueError(f"cfg.pick must be one of union|largest|best_score, got: {cfg.pick}")

        self.model = Sam3Model.from_pretrained(cfg.model_id).to(self.device)
        self.processor = Sam3Processor.from_pretrained(cfg.model_id)
        self.model.eval()

    @torch.no_grad()
    def segment_human_mask(self, image: Image.Image) -> Tuple[Optional[np.ndarray], Optional[float]]:
        """
        Returns:
          (human_mask_bool(H,W) or None, best_score or None)
        """
        inputs = self.processor(images=image, text=self.cfg.prompt, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)

        result = self.processor.post_process_instance_segmentation(
            outputs,
            threshold=self.cfg.threshold,
            mask_threshold=self.cfg.mask_threshold,
            target_sizes=inputs.get("original_sizes").tolist(),
        )[0]

        masks = result.get("masks", None)
        scores = result.get("scores", None)

        if masks is None or len(masks) == 0:
            return None, None

        masks_np = masks.detach().cpu().numpy().astype(bool)  # (N,H,W)

        best_score = None
        if scores is not None and len(scores) > 0:
            scores_np = scores.detach().cpu().numpy().astype(float)
            best_score = float(np.max(scores_np))
        else:
            scores_np = None

        if self.cfg.pick == "union":
            human_mask = np.any(masks_np, axis=0)

        elif self.cfg.pick == "largest":
            areas = masks_np.reshape(masks_np.shape[0], -1).sum(axis=1)
            idx = int(np.argmax(areas))
            human_mask = masks_np[idx]
            if scores_np is not None:
                best_score = float(scores_np[idx])

        else:  # "best_score"
            if scores_np is None:
                human_mask = np.any(masks_np, axis=0)
            else:
                idx = int(np.argmax(scores_np))
                human_mask = masks_np[idx]
                best_score = float(scores_np[idx])

        return human_mask, best_score

    def whiten_background(self, image: Image.Image, human_mask: np.ndarray) -> Image.Image:
        img_np = np.array(image.convert("RGB"))
        human_mask = human_mask.astype(bool)

        out = img_np.copy()
        out[~human_mask] = 255  # white
        return Image.fromarray(out, mode="RGB")

    def make_overlay(self, image_rgb: np.ndarray, human_mask: np.ndarray) -> np.ndarray:
        """
        Returns RGB uint8 overlay (same shape as image).
        Only for W&B debug logging; not saved.
        """
        human_mask = human_mask.astype(bool)
        overlay = image_rgb.copy().astype(np.float32)

        color = np.array(self.cfg.overlay_color_rgb, dtype=np.float32)[None, None, :]
        alpha = float(self.cfg.overlay_alpha)

        # alpha blend only where mask is True
        overlay[human_mask] = (1.0 - alpha) * overlay[human_mask] + alpha * color
        return np.clip(overlay, 0, 255).astype(np.uint8)

    def save_mask_png(self, mask_bool: np.ndarray, masks_dir: Path, stem: str) -> Path:
        masks_dir.mkdir(parents=True, exist_ok=True)
        suffix = self.cfg.mask_suffix
        out_path = masks_dir / f"{stem}{suffix}.png"

        # store as 0/255 grayscale
        mask_u8 = (mask_bool.astype(np.uint8) * 255)
        Image.fromarray(mask_u8, mode="L").save(out_path, format="PNG")
        return out_path

    def process_image_path(
        self,
        img_path: Path,
        masks_dir: Path,
        wandb_run: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        - loads qwen image
        - segments human
        - saves mask into masks_dir
        - whites background and overwrites qwen image
        - logs to wandb if available
        """
        img_path = Path(img_path)
        image = Image.open(img_path).convert("RGB")

        human_mask, best_score = self.segment_human_mask(image)
        if human_mask is None:
            return {"path": str(img_path), "found": False, "saved": False}

        # Save mask
        mask_path = self.save_mask_png(human_mask, masks_dir=masks_dir, stem=img_path.stem)

        # Whiten and overwrite
        out_img = self.whiten_background(image, human_mask)

        if (not self.cfg.overwrite) and img_path.exists():
            save_path = img_path.with_name(img_path.stem + "_bgwhite" + img_path.suffix)
        else:
            save_path = img_path

        out_img.save(save_path, format="PNG" if save_path.suffix.lower() == ".png" else None)

        # W&B logging
        if self.cfg.wandb_log and wandb_run is not None:
            try:
                import wandb  # local import
                img_rgb = np.array(image, dtype=np.uint8)
                out_rgb = np.array(out_img, dtype=np.uint8)
                mask_u8 = (human_mask.astype(np.uint8) * 255)
                mask_rgb = np.stack([mask_u8, mask_u8, mask_u8], axis=-1)
                overlay_rgb = self.make_overlay(img_rgb, human_mask)

                name = img_path.name
                prefix = self.cfg.wandb_prefix.rstrip("/")

                wandb.log({
                    f"{prefix}/segmented": wandb.Image(out_rgb, caption=f"{name} (bg white)"),
                    f"{prefix}/mask": wandb.Image(mask_rgb, caption=f"{name} mask"),
                    f"{prefix}/overlay": wandb.Image(overlay_rgb, caption=f"{name} overlay"),
                    f"{prefix}/human_ratio": float(human_mask.mean()),
                    f"{prefix}/best_score": best_score if best_score is not None else float("nan"),
                })
            except Exception as e:
                # don't fail pipeline due to debug logging
                return {
                    "path": str(save_path),
                    "found": True,
                    "saved": True,
                    "mask_path": str(mask_path),
                    "human_ratio": float(human_mask.mean()),
                    "best_score": best_score,
                    "wandb_error": str(e),
                }

        return {
            "path": str(save_path),
            "found": True,
            "saved": True,
            "mask_path": str(mask_path),
            "human_ratio": float(human_mask.mean()),
            "best_score": best_score,
        }

    def run_on_qwen_dir(
        self,
        scene_dir: Path,
        qwen_images_dir: Path,
        exts: Optional[List[str]] = None,
        wandb_run: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Processes all qwen images, writes masks into <scene_dir>/<masks_dir_name>/,
        overwrites qwen images, logs to wandb.
        """
        scene_dir = Path(scene_dir).resolve()
        qwen_images_dir = Path(qwen_images_dir).resolve()

        if not qwen_images_dir.exists():
            raise FileNotFoundError(f"Missing qwen images dir: {qwen_images_dir}")

        masks_dir = scene_dir / self.cfg.masks_dir_name
        masks_dir.mkdir(parents=True, exist_ok=True)

        if exts is None:
            exts = [".png", ".jpg", ".jpeg"]

        paths = sorted([p for p in qwen_images_dir.iterdir() if p.is_file() and p.suffix.lower() in exts])

        total = len(paths)
        found = 0
        saved = 0
        failures = 0
        details = []

        for p in paths:
            try:
                r = self.process_image_path(p, masks_dir=masks_dir, wandb_run=wandb_run)
                details.append(r)
                if r.get("found"):
                    found += 1
                if r.get("saved"):
                    saved += 1
            except Exception as e:
                failures += 1
                details.append({"path": str(p), "error": str(e)})

        return {
            "scene_dir": str(scene_dir),
            "qwen_images_dir": str(qwen_images_dir),
            "masks_dir": str(masks_dir),
            "total": total,
            "found": found,
            "saved": saved,
            "failures": failures,
            "details": details,
        }
