#vton3d/utils/masked_optical_flow.py
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import cv2
import numpy as np
import torch


def infer_eval_flag_from_path(p: Path) -> str:
    parts = [x.lower() for x in p.parts]
    if "dress" in parts:
        return "dress"
    if "lower" in parts and "upper" in parts:
        raise ValueError(f"Ambiguous path contains both 'upper' and 'lower': {p}")
    if "lower" in parts:
        return "lower"
    if "upper" in parts:
        return "upper"
    raise ValueError(f"Could not infer eval_flag from path (need 'upper'/'lower'/'dress'): {p}")

def infer_length_flag_from_path(p: Path) -> str:
    parts = [x.lower() for x in p.parts]
    if "long" in parts and "short" in parts:
        raise ValueError(f"Ambiguous path contains both 'long' and 'short': {p}")
    if "long" in parts:
        return "long"
    if "short" in parts:
        return "short"
    raise ValueError(f"Could not infer length_flag from path (need 'long'/'short'): {p}")


@dataclass
class MaskedOpticalFlowConfig:
    target_h: int = 1248
    target_w: int = 704

    # Sapiens
    sapiens_repo: str | Path = "Sapiens-Pytorch-Inference"
    sapiens_variant: str = "SEGMENTATION_1B"

    flag_source_path: Optional[str | Path] = None
    clothing_flag: Optional[str] = None
    length_flag: Optional[str] = None

    class_candidates: Optional[Dict[str, List[str]]] = None

    mask_threshold: int = 127
    dilate_px: int = 10
    dilate_shape: int = cv2.MORPH_ELLIPSE

    ecc_n_iter: int = 400
    ecc_eps: float = 1e-7
    ecc_gauss: int = 5
    ecc_border_mode: int = cv2.BORDER_REFLECT

    dis_preset: int = cv2.DISOPTICAL_FLOW_PRESET_MEDIUM
    feather_sigma: float = 7.0
    remap_interp: int = cv2.INTER_CUBIC
    remap_border_mode: int = cv2.BORDER_REFLECT

    def __post_init__(self):
        if self.class_candidates is not None:
            return

        if self.clothing_flag is None:
            if self.flag_source_path is None:
                raise ValueError(
                    "MaskedOpticalFlowConfig: flag_source_path is required (set qwen.clothing_image)."
                )

            else:
                p = Path(self.flag_source_path)
                self.clothing_flag = infer_eval_flag_from_path(p)
                if self.clothing_flag == "dress":
                    self.length_flag = None
                else:
                    if self.length_flag is None:
                        self.length_flag = infer_length_flag_from_path(p)

        flag = self.clothing_flag.lower()
        if flag not in ("upper", "lower", "dress"):
            raise ValueError(f"Invalid clothing_flag='{self.clothing_flag}', expected upper/lower/dress")

        if self.length_flag is not None:
            lf = self.length_flag.lower()
            if lf not in ("short", "long"):
                raise ValueError(f"Invalid length_flag='{self.length_flag}', expected short/long")

        upper = ["Upper Clothing"]
        lower = ["Lower Clothing"]

        arms = {
            "left_upper_arm": ["Left Upper Arm", "Upper Arm Left", "LeftUpperArm"],
            "right_upper_arm": ["Right Upper Arm", "Upper Arm Right", "RightUpperArm"],
            "left_lower_arm": ["Left Lower Arm", "Lower Arm Left", "LeftLowerArm"],
            "right_lower_arm": ["Right Lower Arm", "Lower Arm Right", "RightLowerArm"],
        }
        legs = {
            "left_upper_leg": ["Left Upper Leg", "Upper Leg Left", "LeftUpperLeg"],
            "right_upper_leg": ["Right Upper Leg", "Upper Leg Right", "RightUpperLeg"],
            "left_lower_leg": ["Left Lower Leg", "Lower Leg Left", "LeftLowerLeg"],
            "right_lower_leg": ["Right Lower Leg", "Lower Leg Right", "RightLowerLeg"],
        }

        candidates: Dict[str, List[str]] = {}

        if flag == "upper":
            candidates["upper_clothing"] = upper
            if (self.length_flag or "short").lower() == "long":
                candidates.update(arms)

        elif flag == "lower":
            candidates["lower_clothing"] = lower
            if (self.length_flag or "short").lower() == "long":
                candidates.update(legs)

        else:
            candidates["upper_clothing"] = upper
            candidates["lower_clothing"] = lower
            candidates.update(arms)
            candidates.update(legs)

        self.class_candidates = candidates


class MaskedOpticalFlow:
    """
    Pipeline:
      1) Load src/tgt, resize to target size
      2) Sapiens segment each -> mask (upper clothing + arms)
      3) Union masks -> dilate (wider ignore)
      4) ECC global affine align src->tgt ignoring mask region
      5) DIS residual flow only outside mask (soft edge) and remap
      6) Save aligned image (and optional debug)
    """

    def __init__(self, cfg: MaskedOpticalFlowConfig):
        self.cfg = cfg
        self._estimator = None
        self._classes = None
        self._target_ids = None
        self._init_sapiens_once()

    def run_from_paths(
        self,
        src_path: str | Path,
        tgt_path: str | Path,
        output_path: str | Path,
        debug_dir: Optional[str | Path] = None,
    ) -> Dict:
        src_path = Path(src_path)
        tgt_path = Path(tgt_path)
        output_path = Path(output_path)

        src_bgr = self._load_and_resize_bgr(src_path)
        tgt_bgr = self._load_and_resize_bgr(tgt_path)

        mask_src = self._segment_and_build_mask(src_bgr)
        mask_tgt = self._segment_and_build_mask(tgt_bgr)

        mask_ignore = self._build_union_ignore(mask_tgt, mask_src)

        try:
            src_global, warp_matrix, cc = self._ecc_global_align_affine(
                src_bgr, tgt_bgr, mask_ignore
            )
        except cv2.error as e:
            print("[MaskedOpticalFlow] ECC failed:", e)
            src_global = src_bgr.copy()
            warp_matrix = np.eye(2, 3, dtype=np.float32)
            cc = None

        aligned, flow = self._residual_flow_warp(src_global, tgt_bgr, mask_ignore)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        ok = cv2.imwrite(str(output_path), aligned)
        if not ok:
            raise IOError(f"cv2.imwrite failed for: {output_path}")

        info = {
            "output_path": str(output_path),
            "warp_matrix": warp_matrix,
            "ecc_cc": cc,
            "flow": flow,
            "mask_ignore": mask_ignore,
            "mask_src": mask_src,
            "mask_tgt": mask_tgt,
        }

        if debug_dir is not None:
            self._dump_debug(debug_dir, info, aligned)

        return info

    def _init_sapiens_once(self):
        repo = Path(self.cfg.sapiens_repo).resolve()
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))

        from sapiens_inference.segmentation import (
            SapiensSegmentation,
            SapiensSegmentationType,
            classes,
        )

        self._classes = classes

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if device.type == "cuda" else torch.float32

        seg_type = getattr(SapiensSegmentationType, self.cfg.sapiens_variant)

        self._estimator = SapiensSegmentation(seg_type, device=device, dtype=dtype)

        target_ids = []
        for _, candidates in self.cfg.class_candidates.items():
            target_ids.append(self._find_class_id_any(candidates, self._classes))
        self._target_ids = np.array(target_ids, dtype=np.int32)

        print(f"[MaskedOpticalFlow] clothing_flag={self.cfg.clothing_flag}, length_flag={self.cfg.length_flag}")
        print("[MaskedOpticalFlow] Using class IDs:")
        for cid in self._target_ids:
            print(f"  - {cid:3d} | {self._classes[cid]}")

    @staticmethod
    def _find_class_id_any(name_candidates: List[str], classes_list: List[str]) -> int:
        for cand in name_candidates:
            if cand in classes_list:
                return classes_list.index(cand)

        lower_map = {n.lower(): i for i, n in enumerate(classes_list)}
        for cand in name_candidates:
            key = cand.lower()
            if key in lower_map:
                return lower_map[key]

        for cand in name_candidates:
            cl = cand.lower()
            for i, n in enumerate(classes_list):
                if cl in n.lower():
                    return i

        raise ValueError(
            f"Could not find class for candidates: {name_candidates}\n"
            f"Available arm/cloth-ish: {[n for n in classes_list if ('arm' in n.lower() or 'cloth' in n.lower())]}"
        )

    def _segment_and_build_mask(self, img_bgr: np.ndarray) -> np.ndarray:
        seg_map = self._estimator(img_bgr).astype(np.int32)
        mask = np.isin(seg_map, self._target_ids).astype(np.uint8) * 255
        return mask

    def _load_and_resize_bgr(self, path: Path) -> np.ndarray:
        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(f"Could not load image: {path}")

        h, w = img.shape[:2]
        th, tw = self.cfg.target_h, self.cfg.target_w
        if (h, w) != (th, tw):
            interp = cv2.INTER_AREA if (h > th or w > tw) else cv2.INTER_LINEAR
            img = cv2.resize(img, (tw, th), interpolation=interp)

        return img

    def _build_union_ignore(self, mask1: np.ndarray, mask2: np.ndarray) -> np.ndarray:
        # ensure binary
        _, m1 = cv2.threshold(mask1, self.cfg.mask_threshold, 255, cv2.THRESH_BINARY)
        _, m2 = cv2.threshold(mask2, self.cfg.mask_threshold, 255, cv2.THRESH_BINARY)

        union = cv2.bitwise_or(m1, m2)

        if self.cfg.dilate_px and self.cfg.dilate_px > 0:
            ksz = 2 * self.cfg.dilate_px + 1
            kernel = cv2.getStructuringElement(self.cfg.dilate_shape, (ksz, ksz))
            union = cv2.dilate(union, kernel)

        return union

    def _ecc_global_align_affine(
        self,
        src_bgr: np.ndarray,
        tgt_bgr: np.ndarray,
        mask_ignore_tgt: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, float]:

        h, w = tgt_bgr.shape[:2]
        if src_bgr.shape[:2] != (h, w):
            src_bgr = cv2.resize(src_bgr, (w, h), interpolation=cv2.INTER_LINEAR)

        src_g = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2GRAY)
        tgt_g = cv2.cvtColor(tgt_bgr, cv2.COLOR_BGR2GRAY)

        ecc_mask = (mask_ignore_tgt == 0).astype(np.uint8) * 255

        warp_mode = cv2.MOTION_AFFINE
        warp_matrix = np.eye(2, 3, dtype=np.float32)

        criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            self.cfg.ecc_n_iter,
            self.cfg.ecc_eps,
        )

        cc, warp_matrix = cv2.findTransformECC(
            templateImage=tgt_g,
            inputImage=src_g,
            warpMatrix=warp_matrix,
            motionType=warp_mode,
            criteria=criteria,
            inputMask=ecc_mask,
            gaussFiltSize=self.cfg.ecc_gauss,
        )

        src_global = cv2.warpAffine(
            src_bgr,
            warp_matrix,
            (w, h),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=self.cfg.ecc_border_mode,
        )

        return src_global, warp_matrix, cc


    def _residual_flow_warp(
        self,
        src_bgr_global: np.ndarray,
        tgt_bgr: np.ndarray,
        mask_ignore_tgt: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:

        h, w = tgt_bgr.shape[:2]

        src_g = cv2.cvtColor(src_bgr_global, cv2.COLOR_BGR2GRAY)
        tgt_g = cv2.cvtColor(tgt_bgr, cv2.COLOR_BGR2GRAY)

        valid = (mask_ignore_tgt == 0).astype(np.float32)

        def grad(img):
            gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
            mag = cv2.magnitude(gx, gy)
            mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            return mag.astype(np.uint8)

        src_for_flow = grad(src_g)
        tgt_for_flow = grad(tgt_g)

        dis = cv2.DISOpticalFlow_create(self.cfg.dis_preset)
        flow = dis.calc(tgt_for_flow, src_for_flow, None).astype(np.float32)

        valid_soft = cv2.GaussianBlur(valid, (0, 0), self.cfg.feather_sigma)
        flow = flow * valid_soft[..., None]

        grid_x, grid_y = np.meshgrid(
            np.arange(w, dtype=np.float32),
            np.arange(h, dtype=np.float32),
        )
        map_x = grid_x + flow[..., 0]
        map_y = grid_y + flow[..., 1]

        warped = cv2.remap(
            src_bgr_global,
            map_x,
            map_y,
            interpolation=self.cfg.remap_interp,
            borderMode=self.cfg.remap_border_mode,
        )

        return warped, flow

    #debug:

    def _dump_debug(self, debug_dir: str | Path, info: Dict, aligned: np.ndarray):
        d = Path(debug_dir)
        d.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(d / "mask_src.png"), info["mask_src"])
        cv2.imwrite(str(d / "mask_tgt.png"), info["mask_tgt"])
        cv2.imwrite(str(d / "mask_ignore_union_dilated.png"), info["mask_ignore"])
        cv2.imwrite(str(d / "aligned.png"), aligned)

        flow = info.get("flow", None)
        if flow is not None:
            mag = np.linalg.norm(flow, axis=2)
            mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            cv2.imwrite(str(d / "flow_mag.png"), mag.astype(np.uint8))
