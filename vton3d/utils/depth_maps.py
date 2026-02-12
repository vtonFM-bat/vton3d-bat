from pathlib import Path
import sys
import os
import cv2
import numpy as np
import torch
from PIL import Image


class SapiensDepthGenerator:
    """
    Generates Sapiens depth maps using external foreground masks.
    Output depth maps are directly compatible with scale-invariant log depth loss.
    """

    def __init__(self, repo_root: Path, device: str = None, depth_type: str = "DEPTH_1B"):
        """
        repo_root: root directory that contains "Sapiens-Pytorch-Inference"
        """

        self.repo_root = Path(repo_root)
        self.sapiens_repo = self.repo_root / "Sapiens-Pytorch-Inference"

        if not self.sapiens_repo.exists():
            raise FileNotFoundError(f"Sapiens repo not found: {self.sapiens_repo}")

        sys.path.insert(0, str(self.sapiens_repo))

        from sapiens_inference import (
            SapiensDepth,
            SapiensDepthType,
            SapiensConfig,
        )

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        cfg = SapiensConfig()
        cfg.depth_type = getattr(SapiensDepthType, depth_type)
        cfg.device = device

        orig_cwd = os.getcwd()
        try:
            os.chdir(self.sapiens_repo)
            self.depth_model = SapiensDepth(cfg.depth_type, cfg.device, cfg.dtype)
        finally:
            os.chdir(orig_cwd)

        self.device = device

    def _load_image_bgr(self, path: Path):
        pil = Image.open(path).convert("RGB")
        rgb = np.array(pil)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        return bgr

    def _predict_depth(self, img_path: Path):
        bgr = self._load_image_bgr(img_path)
        H, W = bgr.shape[:2]

        depth = self.depth_model(bgr)

        if isinstance(depth, torch.Tensor):
            depth = depth.squeeze().detach().cpu().numpy()
        else:
            depth = np.squeeze(depth)

        if depth.shape != (H, W):
            depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_LINEAR)

        return depth.astype(np.float32)

    @staticmethod
    def _postprocess_for_si_loss(depth, mask, eps=1e-3):
        """
        Makes depth compatible with scale-invariant log loss:
        - background -> NaN
        - remove non-finite
        - ensure strictly positive
        - median normalization (robust scaling)
        """

        Z = depth.astype(np.float32)

        # Remove invalid values
        Z[~np.isfinite(Z)] = np.nan

        # Apply foreground mask (0 = background)
        Z[mask == 0] = np.nan

        valid = np.isfinite(Z)
        if not valid.any():
            return Z

        # Shift if necessary to ensure positivity
        zmin = np.nanmin(Z)
        if zmin <= 0:
            Z = Z - zmin + eps

        Z[(np.isfinite(Z)) & (Z <= 0)] = eps

        # Median scaling (important for stable training)
        valid = np.isfinite(Z)
        median = np.nanmedian(Z[valid])
        if median > 0:
            Z = Z / median

        return Z.astype(np.float32)

    def generate_depth_folder(
        self,
        input_dir: str,
        mask_dir: str,
        output_dir: str,
        image_exts=(".jpg", ".jpeg", ".png"),
        mask_ext=".png",
        overwrite=False,
    ):
        """
        Generates depth maps for all images in input_dir.
        Mask files must exist in mask_dir with identical relative paths.
        """

        input_dir = Path(input_dir)
        mask_dir = Path(mask_dir)
        output_dir = Path(output_dir)

        images = [p for p in input_dir.rglob("*") if p.suffix.lower() in image_exts]

        for img_path in images:
            rel = img_path.relative_to(input_dir)
            stem = rel.with_suffix("")

            mask_path = mask_dir / (str(stem) + mask_ext)
            out_path = output_dir / (str(stem) + ".npy")

            if out_path.exists() and not overwrite:
                continue

            if not mask_path.exists():
                raise FileNotFoundError(f"Mask missing: {mask_path}")

            depth_raw = self._predict_depth(img_path)

            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise FileNotFoundError(f"Could not load mask: {mask_path}")

            if mask.shape != depth_raw.shape:
                mask = cv2.resize(mask, depth_raw.shape[::-1], interpolation=cv2.INTER_NEAREST)

            mask = (mask > 127).astype(np.uint8)

            depth_processed = self._postprocess_for_si_loss(depth_raw, mask)

            out_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(out_path, depth_processed)