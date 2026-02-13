# depth_map.py
from pathlib import Path
import sys
import os
import cv2
import numpy as np
import torch
from PIL import Image


class SapiensDepthGenerator:
    """
    Generate RAW Sapiens depth maps (no normalization / no shifting / no masking).

    - Output is float32 npy, same HxW as the input image.
    - If a mask is provided, it is used ONLY to mark invalid/background pixels as NaN
      (optional; can be disabled).
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

        from sapiens_inference import SapiensDepth, SapiensDepthType, SapiensConfig

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

    @staticmethod
    def _load_image_bgr(path: Path) -> np.ndarray:
        pil = Image.open(path).convert("RGB")
        rgb = np.array(pil)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        return bgr

    def _predict_depth_raw(self, img_path: Path) -> np.ndarray:
        """Returns raw Sapiens output resized to the input image resolution."""
        bgr = self._load_image_bgr(img_path)
        H, W = bgr.shape[:2]

        depth = self.depth_model(bgr)

        if isinstance(depth, torch.Tensor):
            depth = depth.squeeze().detach().cpu().numpy()
        else:
            depth = np.squeeze(depth)

        # Ensure shape matches the image
        if depth.shape != (H, W):
            depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_LINEAR)

        return depth.astype(np.float32)

    @staticmethod
    def _load_mask(mask_path: Path, target_hw: tuple[int, int]) -> np.ndarray:
        """Load mask as uint8 {0,1} resized with nearest-neighbor to target (H,W)."""
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Could not load mask: {mask_path}")
        H, W = target_hw
        if mask.shape != (H, W):
            mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 127).astype(np.uint8)
        return mask

    @staticmethod
    def _apply_mask_as_far(depth: np.ndarray, mask01: np.ndarray) -> np.ndarray:
        """
        Set background pixels to a far distance instead of NaN.
        This stabilizes Gaussian Splatting by preventing floaters.
        """
        out = depth.astype(np.float32, copy=True)

        valid = np.isfinite(out) & (mask01 > 0)
        if not np.any(valid):
            return out

        # robust far plane estimate
        far_val = np.percentile(out[valid], 99.5) * 1.5

        out[mask01 == 0] = far_val
        return out

    def generate_depth_folder(
        self,
        input_dir: str,
        output_dir: str,
        image_exts=(".jpg", ".jpeg", ".png"),
        overwrite: bool = False,
        mask_dir: str | None = None,
        mask_ext: str = ".png",
        apply_mask_to_depth: bool = True,
    ):
        """
        Generates raw depth maps for all images in input_dir.

        If mask_dir is provided and apply_mask_to_depth=True:
          - background pixels are set to NaN in the saved depth map.

        Naming:
          - Keeps relative paths.
          - Saves as <output_dir>/<rel_path_without_ext>.npy
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        mask_dir_path = Path(mask_dir) if mask_dir is not None else None

        images = [p for p in input_dir.rglob("*") if p.suffix.lower() in image_exts]

        for img_path in images:
            rel = img_path.relative_to(input_dir)
            stem = rel.with_suffix("")
            out_path = output_dir / (str(stem) + ".npy")

            if out_path.exists() and not overwrite:
                continue

            depth_raw = self._predict_depth_raw(img_path)

            if mask_dir_path is not None and apply_mask_to_depth:
                mask_path = mask_dir_path / (str(stem) + mask_ext)
                if not mask_path.exists():
                    raise FileNotFoundError(f"Mask missing: {mask_path}")
                mask01 = self._load_mask(mask_path, target_hw=depth_raw.shape)
                depth_raw = self._apply_mask_as_far(depth_raw, mask01)

            out_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(out_path, depth_raw)
