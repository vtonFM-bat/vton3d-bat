#vton3d/utils/qwen_eval.py
import os
import sys
import argparse
import cv2
import numpy as np
from PIL import Image
import torch
import cv2
import numpy as np
import matplotlib.cm as cm

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
SAPIENS_REPO = REPO_ROOT / "Sapiens-Pytorch-Inference"
sys.path.insert(0, str(SAPIENS_REPO))

from sapiens_inference.segmentation import classes

from transformers import CLIPProcessor, CLIPModel


def get_clothing_class_idx(flag: str) -> int:
    """
    Returns the Sapiens class index for "upper" or "lower" clothing.
    """
    flag = flag.lower()
    if flag == "upper":
        target = "Upper Clothing"
    elif flag == "lower":
        target = "Lower Clothing"
    else:
        raise ValueError(f"Invalid flag '{flag}'. Expected 'upper' or 'lower'.")

    if target not in classes:
        raise ValueError(f"Class '{target}' not found in Sapiens class list.")

    return classes.index(target)


def _idx(name: str):
    try:
        return classes.index(name)
    except ValueError:
        return None

def compute_psnr_from_mse(mse_value: float, max_val: float = 1.0) -> float:
    """
    Computes PSNR from a given MSE value and maximum pixel value.
    """
    if mse_value <= 0.0:
        return float("inf")
    return 10.0 * np.log10((max_val ** 2) / mse_value)

def qwen_eval_masked(img1_path, img2_path, flag, length_flag, estimator):
    """
    Loads two images, segments the first image, excludes a clothing class
    (upper/lower) or (dress = upper+lower+arms+legs), computes MSE only on
    unmasked pixels, and returns both the MSE value and a masked difference heatmap.

    Dress behavior:
      - mask out: Upper Clothing + Lower Clothing
      - mask out: Left/Right Upper/Lower Arm
      - mask out: Left/Right Upper/Lower Leg
      - length_flag can be None for dress
    """
    img1_bgr = cv2.imread(img1_path)
    img2_bgr = cv2.imread(img2_path)

    if img1_bgr is None:
        raise FileNotFoundError(f"Could not load image: {img1_path}")
    if img2_bgr is None:
        raise FileNotFoundError(f"Could not load image: {img2_path}")

    if img1_bgr.shape != img2_bgr.shape:
        raise ValueError("Images must have the same shape.")

    seg_map = estimator(img1_bgr).astype(np.int32)

    flag = flag.lower()
    if flag not in ("upper", "lower", "dress"):
        raise ValueError(f"Invalid flag '{flag}', expected 'upper', 'lower', or 'dress'.")

    if length_flag is not None:
        length_flag = length_flag.lower()
        if length_flag not in ("short", "long"):
            raise ValueError(f"Invalid length_flag '{length_flag}', expected 'short' or 'long'.")

    # Build exclusion mask
    if flag in ("upper", "lower"):
        clothing_name = "Upper Clothing" if flag == "upper" else "Lower Clothing"
        clothing_idx = _idx(clothing_name)
        if clothing_idx is None:
            raise ValueError(f"Class '{clothing_name}' not found in classes list.")

        mask_exclude = (seg_map == clothing_idx)

        # Existing long/short behavior: additionally exclude arms/legs if long
        if length_flag == "long":
            if flag == "upper":
                arm_names = [
                    "Left Upper Arm", "Right Upper Arm",
                    "Left Lower Arm", "Right Lower Arm",
                ]
                arm_indices = [i for i in (_idx(n) for n in arm_names) if i is not None]
                if any(np.any(seg_map == ai) for ai in arm_indices):
                    for ai in arm_indices:
                        mask_exclude |= (seg_map == ai)
            else:  # lower
                leg_names = [
                    "Left Upper Leg", "Right Upper Leg",
                    "Left Lower Leg", "Right Lower Leg",
                ]
                leg_indices = [i for i in (_idx(n) for n in leg_names) if i is not None]
                if any(np.any(seg_map == li) for li in leg_indices):
                    for li in leg_indices:
                        mask_exclude |= (seg_map == li)

    else:
        # dress: exclude Upper+Lower clothing + ALL arms + ALL legs (always)
        exclude_names = [
            "Upper Clothing",
            "Lower Clothing",
            "Left Upper Arm", "Right Upper Arm",
            "Left Lower Arm", "Right Lower Arm",
            "Left Upper Leg", "Right Upper Leg",
            "Left Lower Leg", "Right Lower Leg",
        ]
        exclude_indices = [i for i in (_idx(n) for n in exclude_names) if i is not None]
        if not exclude_indices:
            raise ValueError("No dress-related classes found in classes list.")

        mask_exclude = np.isin(seg_map, exclude_indices)

    mask_include = ~mask_exclude

    # Compute masked MSE/PSNR (on included pixels only)
    img1 = img1_bgr.astype(np.float32) / 255.0
    img2 = img2_bgr.astype(np.float32) / 255.0

    diff = (img1 - img2) ** 2
    diff_masked = diff[mask_include]
    mse_value = diff_masked.mean().item()
    psnr_value = compute_psnr_from_mse(mse_value, max_val=1.0)

    # Heatmap: average abs diff per pixel, with excluded region set to 0
    abs_diff = np.abs(img1 - img2)
    heatmap_gray = abs_diff.mean(axis=2)
    heatmap_gray[mask_exclude] = 0.0

    norm = heatmap_gray / (heatmap_gray.max() + 1e-8)
    colormap = cm.get_cmap("Reds")
    heatmap_red = (colormap(norm)[..., :3] * 255).astype(np.uint8)
    heatmap_red = cv2.cvtColor(heatmap_red, cv2.COLOR_RGB2BGR)

    return mse_value, psnr_value, heatmap_red


#fashionclip
_FC_CACHE = {
    "device": None,
    "processor": None,
    "model": None,
    "ref_path": None,
    "ref_emb": None,
}

def qwen_fashionclip_similarity_masked_clothing(
    person_img_path: str,
    clothing_ref_path: str,
    flag: str,
    estimator,
    clip_device: str = "cpu",
    return_masked_rgb: bool = False,
):
    """
    Segments ONLY the clothing class (Upper/Lower) from the PERSON image,
    makes everything else white, embeds with Fashion-CLIP, embeds the clothing
    reference image, returns cosine similarity.

    Returns:
      sim: float
      masked_rgb (optional): np.ndarray HxWx3 uint8 (RGB) if return_masked_rgb=True else None
    """
    flag = flag.lower()
    if flag not in ("upper", "lower",  "dress"):
        raise ValueError(f"Invalid flag '{flag}', expected 'upper',  'dress' or 'lower'.")

    if flag == "dress":
        clothing_name = "Upper Clothing"
    else:
        clothing_name = "Upper Clothing" if flag == "upper" else "Lower Clothing"
    if clothing_name not in classes:
        raise ValueError(f"Class '{clothing_name}' not found in classes list.")
    clothing_idx = classes.index(clothing_name)

    img_bgr = cv2.imread(person_img_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not load image: {person_img_path}")

    seg_map = estimator(img_bgr).astype(np.int32)
    mask = (seg_map == clothing_idx)

    white_bgr = img_bgr.copy()
    white_bgr[~mask] = (255, 255, 255)

    white_rgb = white_bgr[..., ::-1]
    person_pil = Image.fromarray(white_rgb)

    ref_pil = Image.open(clothing_ref_path).convert("RGB")

    dev = torch.device(clip_device)

    if _FC_CACHE["processor"] is None or _FC_CACHE["model"] is None or _FC_CACHE["device"] != clip_device:
        clip_id = "patrickjohncyh/fashion-clip"
        _FC_CACHE["processor"] = CLIPProcessor.from_pretrained(clip_id, use_fast=True)
        _FC_CACHE["model"] = CLIPModel.from_pretrained(clip_id).to(dev)
        _FC_CACHE["model"].eval()
        _FC_CACHE["device"] = clip_device
        _FC_CACHE["ref_path"] = None
        _FC_CACHE["ref_emb"] = None

    processor = _FC_CACHE["processor"]
    model = _FC_CACHE["model"]

    def embed(pil_img: Image.Image) -> torch.Tensor:
        inputs = processor(images=pil_img, return_tensors="pt")
        inputs = {k: v.to(dev) for k, v in inputs.items()}

        with torch.inference_mode():
            out = model.get_image_features(pixel_values=inputs["pixel_values"])

        if isinstance(out, torch.Tensor):
            feats = out
        elif hasattr(out, "image_embeds") and out.image_embeds is not None:
            feats = out.image_embeds
        elif hasattr(out, "pooler_output") and out.pooler_output is not None:
            feats = out.pooler_output
        elif hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
            # CLS token fallback
            feats = out.last_hidden_state[:, 0, :]
        else:
            raise RuntimeError(f"Unexpected output type from get_image_features: {type(out)}")

        feats = feats.float()
        feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-12)
        return feats.squeeze(0)

    if _FC_CACHE["ref_path"] != clothing_ref_path or _FC_CACHE["ref_emb"] is None:
        _FC_CACHE["ref_emb"] = embed(ref_pil)
        _FC_CACHE["ref_path"] = clothing_ref_path

    emb_person = embed(person_pil)
    emb_ref = _FC_CACHE["ref_emb"]

    sim = float(torch.dot(emb_person, emb_ref).detach().cpu().item())

    masked_rgb = white_rgb if return_masked_rgb else None
    return sim, masked_rgb

_ARCFACE_CACHE = {
    "app": None,
    "ctx_id": None,
    "det_size": None,
}

def qwen_arcface_similarity_input_vs_output(
    img1_path: str,
    img2_path: str,
    device: str = "cpu",                 # "cpu" or "cuda"
    det_size: tuple[int, int] = (640, 640),
    return_faces_rgb: bool = False,
):
    """
    ArcFace cosine similarity between faces in img1 and img2 (input vs output).
    If no face is detected in one of the images, returns np.nan (NA) and no crops.

    Returns:
      sim: float (cosine) or np.nan
      face1_rgb: np.ndarray (RGB) or None
      face2_rgb: np.ndarray (RGB) or None
    """

    from insightface.app import FaceAnalysis

    img1_bgr = cv2.imread(img1_path)
    img2_bgr = cv2.imread(img2_path)
    if img1_bgr is None or img2_bgr is None:
        return (np.nan, None, None) if return_faces_rgb else (np.nan,)

    ctx_id = 0 if device.lower() in ("cuda", "gpu") else -1

    if (
        _ARCFACE_CACHE["app"] is None
        or _ARCFACE_CACHE["ctx_id"] != ctx_id
        or _ARCFACE_CACHE["det_size"] != det_size
    ):
        app = FaceAnalysis(name="buffalo_l")
        app.prepare(ctx_id=ctx_id, det_size=det_size)
        _ARCFACE_CACHE["app"] = app
        _ARCFACE_CACHE["ctx_id"] = ctx_id
        _ARCFACE_CACHE["det_size"] = det_size

    app = _ARCFACE_CACHE["app"]

    faces1 = app.get(img1_bgr)
    faces2 = app.get(img2_bgr)

    if len(faces1) == 0 or len(faces2) == 0:
        return (np.nan, None, None) if return_faces_rgb else (np.nan,)

    def pick_largest(faces):
        areas = []
        for f in faces:
            x1, y1, x2, y2 = f.bbox
            areas.append((x2 - x1) * (y2 - y1))
        return faces[int(np.argmax(areas))]

    f1 = pick_largest(faces1)
    f2 = pick_largest(faces2)

    emb1 = f1.normed_embedding if hasattr(f1, "normed_embedding") else f1.embedding
    emb2 = f2.normed_embedding if hasattr(f2, "normed_embedding") else f2.embedding

    emb1 = emb1 / (np.linalg.norm(emb1) + 1e-12)
    emb2 = emb2 / (np.linalg.norm(emb2) + 1e-12)

    sim = float(np.dot(emb1, emb2))

    if not return_faces_rgb:
        return (sim,)

    def face_rgb(img_bgr, face):
        if hasattr(face, "crop") and face.crop is not None:
            crop_bgr = face.crop
        else:
            x1, y1, x2, y2 = face.bbox.astype(int)
            h, w = img_bgr.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            crop_bgr = img_bgr[y1:y2, x1:x2]
        return crop_bgr[..., ::-1]

    face1_rgb = face_rgb(img1_bgr, f1)
    face2_rgb = face_rgb(img2_bgr, f2)

    return sim, face1_rgb, face2_rgb