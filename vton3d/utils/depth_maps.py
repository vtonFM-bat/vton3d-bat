import os
import sys
from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np
import torch
from PIL import Image
REPO_ROOT = Path(__file__).resolve().parents[2]
SAPIENS_REPO = REPO_ROOT / "Sapiens-Pytorch-Inference"
sys.path.insert(0, str(SAPIENS_REPO))

from sapiens_inference import (
    SapiensConfig,
    SapiensDepth,
    SapiensDepthType,
)

class SapiensDepthMapBuilder:
    """
    Build per-image depth maps using Sapiens depth inference and external human masks.

    Inputs:
        - RGB images under: data_dir/images (or a custom images_subdir)
        - Human masks under: data_dir/human_mask
          Mask file naming: <image_stem><human_mask_suffix>.png
          Example: frame_0001 + human_masks + .png -> frame_0001human_masks.png

    Outputs:
        - Depth maps under: data_dir/depth_maps (or a custom out_subdir)
        - Output filenames match the original RGB filenames (same stem and extension),
          preserving subfolder structure.

    Depth processing:
        - Sapiens depth is min/max normalized to [0,1] per frame
        - Background pixels (mask==False) are set to far depth (1.0)
        - Optional inversion: depth <- 1-depth

    Save format:
        - save_mode="u16" writes 16-bit PNG (0..65535)
        - save_mode="u8" writes 8-bit PNG (0..255)
    """

    def __init__(
        self,
        sapiens_dir: Path,
        device: Optional[str] = None,
    ) -> None:
        self.sapiens_dir = Path(sapiens_dir)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._depth_predictor = None

    def _ensure_predictor(self) -> None:
        """
        Lazily initialize the Sapiens depth predictor.
        """
        if self._depth_predictor is not None:
            return

        sys.path.insert(0, str(self.sapiens_dir))


        cfg = SapiensConfig()
        cfg.depth_type = SapiensDepthType.DEPTH_1B
        cfg.device = self.device

        orig_cwd = os.getcwd()
        try:
            os.makedirs(self.sapiens_dir / "models", exist_ok=True)
            os.chdir(str(self.sapiens_dir))
            self._depth_predictor = SapiensDepth(cfg.depth_type, cfg.device, cfg.dtype)
        finally:
            os.chdir(orig_cwd)

    @staticmethod
    def _read_rgb_image(path: Path) -> np.ndarray:
        """
        Read an image as RGB uint8 array [H,W,3].
        """
        return np.array(Image.open(path).convert("RGB"))

    @staticmethod
    def _load_mask(mask_path: Path) -> np.ndarray:
        """
        Load a human mask as boolean array [H,W] where white means human and black means background.
        """
        m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if m is None:
            raise FileNotFoundError(mask_path)
        return m > 127

    @staticmethod
    def _normalize_depth(depth: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        """
        Normalize a depth map to [0,1] using robust min/max normalization.
        """
        d = depth.astype(np.float32)
        if not np.isfinite(d).any():
            return np.ones_like(d, dtype=np.float32)

        dmin = np.nanmin(d)
        dmax = np.nanmax(d)

        if not np.isfinite(dmin) or not np.isfinite(dmax) or abs(dmax - dmin) < eps:
            return np.ones_like(d, dtype=np.float32)

        out = (d - dmin) / (dmax - dmin + eps)
        return np.clip(out, 0.0, 1.0)

    def iter_image_paths(
        self,
        data_dir: Path,
        images_subdir: str = "images",
        exts: Iterable[str] = (".png", ".jpg", ".jpeg"),
    ) -> list[Path]:
        """
        Collect and return all image paths under data_dir/images_subdir with given extensions.
        """
        img_dir = Path(data_dir) / images_subdir
        paths = [p for p in img_dir.rglob("*") if p.suffix.lower() in set(exts)]
        paths.sort()
        return paths

    def build_for_dataset(
        self,
        data_dir: Path,
        images_subdir: str = "images",
        human_mask_subdir: str = "human_masks",
        human_mask_suffix: str = "human_masks",
        out_subdir: str = "depth_maps",
        invert_depth: bool = False,
        save_mode: str = "u16",
        skip_existing: bool = True,
        fail_on_missing_mask: bool = False,
    ) -> None:
        """
        Build depth maps for all images in the dataset folder.

        If skip_existing is True, existing outputs will not be recomputed.
        If fail_on_missing_mask is True, missing masks raise an exception.
        """
        self._ensure_predictor()

        data_dir = Path(data_dir)
        img_dir = data_dir / images_subdir
        mask_dir = data_dir / human_mask_subdir
        out_dir = data_dir / out_subdir
        out_dir.mkdir(parents=True, exist_ok=True)

        image_paths = self.iter_image_paths(data_dir=data_dir, images_subdir=images_subdir)

        for img_path in image_paths:
            rel = img_path.relative_to(img_dir)
            out_path = (out_dir / f"{img_path.stem}.png")

            out_path.parent.mkdir(parents=True, exist_ok=True)

            if skip_existing and out_path.exists():
                continue

            base = img_path.stem
            mask_path = mask_dir / f"{base}{human_mask_suffix}.png"

            if not mask_path.exists():
                if fail_on_missing_mask:
                    raise FileNotFoundError(mask_path)
                continue

            rgb = self._read_rgb_image(img_path)
            H, W = rgb.shape[:2]
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            human_mask = self._load_mask(mask_path)
            if human_mask.shape != (H, W):
                human_mask = cv2.resize(
                    human_mask.astype(np.uint8),
                    (W, H),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)

            depth_raw = self._depth_predictor(bgr)
            if isinstance(depth_raw, torch.Tensor):
                depth_np = depth_raw.squeeze().detach().cpu().numpy()
            else:
                depth_np = np.squeeze(depth_raw)

            if depth_np.shape != (H, W):
                depth_np = cv2.resize(depth_np, (W, H), interpolation=cv2.INTER_LINEAR)

            depth_norm = self._normalize_depth(depth_np)
            if invert_depth:
                depth_norm = 1.0 - depth_norm

            depth_bg_far = np.where(human_mask, depth_norm, 1.0)

            if save_mode == "u16":
                out_img = (depth_bg_far * 65535.0).astype(np.uint16)
            elif save_mode == "u8":
                out_img = (depth_bg_far * 255.0).astype(np.uint8)
            else:
                raise ValueError("save_mode must be 'u8' or 'u16'")

            cv2.imwrite(str(out_path), out_img)

    def build_one(
        self,
        image_path: Path,
        mask_path: Path,
        out_path: Path,
        invert_depth: bool = False,
        save_mode: str = "u16",
    ) -> None:
        """
        Build a depth map for a single image using a provided mask path and output path.
        """
        self._ensure_predictor()

        rgb = self._read_rgb_image(Path(image_path))
        H, W = rgb.shape[:2]
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        human_mask = self._load_mask(Path(mask_path))
        if human_mask.shape != (H, W):
            human_mask = cv2.resize(
                human_mask.astype(np.uint8),
                (W, H),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)

        depth_raw = self._depth_predictor(bgr)
        if isinstance(depth_raw, torch.Tensor):
            depth_np = depth_raw.squeeze().detach().cpu().numpy()
        else:
            depth_np = np.squeeze(depth_raw)

        if depth_np.shape != (H, W):
            depth_np = cv2.resize(depth_np, (W, H), interpolation=cv2.INTER_LINEAR)

        depth_norm = self._normalize_depth(depth_np)
        if invert_depth:
            depth_norm = 1.0 - depth_norm

        depth_bg_far = np.where(human_mask, depth_norm, 1.0)

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if save_mode == "u16":
            out_img = (depth_bg_far * 65535.0).astype(np.uint16)
        elif save_mode == "u8":
            out_img = (depth_bg_far * 255.0).astype(np.uint8)
        else:
            raise ValueError("save_mode must be 'u8' or 'u16'")

        cv2.imwrite(str(out_path), out_img)
