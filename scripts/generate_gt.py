#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image

import torch
import wandb
from diffusers import QwenImageEditPlusPipeline

# Uses your exact class/behavior (Sapiens init once, union mask, ECC+DIS).
# Make sure this import path matches your repo.
from vton3d.utils.masked_optical_flow import MaskedOpticalFlow, MaskedOpticalFlowConfig


NEGATIVE_PROMPT = "change pose"


# -----------------------------
# Helpers
# -----------------------------
def is_image_file(p: Path) -> bool:
    return p.suffix.lower() in (".png", ".jpg", ".jpeg")


def is_flat(p: Path) -> bool:
    return "flat" in p.name.lower()


def load_qwen_pipe(model_path: str) -> QwenImageEditPlusPipeline:
    """
    Same style as your snippet: bf16 + cpu offload.
    """
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
    )
    pipe.enable_model_cpu_offload()
    return pipe


def qwen_edit_one(
    pipe: QwenImageEditPlusPipeline,
    person_img: Image.Image,
    garment_word: str,
    true_cfg_scale: float,
    num_inference_steps: int,
    width: int,
    height: int,
    seed: Optional[int] = None,
) -> Image.Image:
    """
    Single-image edit (NO clothing reference image).
    """
    prompt = (
        f"Replace the person's original {garment_word} with a different {garment_word}. "
        "Keep background, pose, face, hair, body shape and all other clothes of the person exactly the same. "
        "Do not invent new accessories or change the scene."
    )

    gen = None
    if seed is not None:
        gen = torch.Generator(device="cpu").manual_seed(int(seed))

    out = pipe(
        image=person_img.convert("RGB"),
        prompt=prompt,
        negative_prompt=NEGATIVE_PROMPT,
        true_cfg_scale=true_cfg_scale,
        num_inference_steps=num_inference_steps,
        generator=gen,
        width=width,
        height=height,
    )
    return out.images[0].convert("RGB")


def bgr_to_wandb_image(bgr: np.ndarray, caption: str) -> wandb.Image:
    rgb = bgr[..., ::-1]
    return wandb.Image(rgb, caption=caption)


def gray_u8_to_wandb_image(gray_u8: np.ndarray, caption: str) -> wandb.Image:
    return wandb.Image(gray_u8, caption=caption)


def diff_heatmap_u8(orig_bgr: np.ndarray, out_bgr: np.ndarray) -> np.ndarray:
    """
    uint8 heatmap (H,W), based on L2 diff in RGB.
    Robust normalization by 99th percentile.
    """
    o = orig_bgr[..., ::-1].astype(np.float32)
    x = out_bgr[..., ::-1].astype(np.float32)
    d = o - x
    mag = np.sqrt(np.sum(d * d, axis=2))  # (H,W)

    p99 = float(np.percentile(mag, 99.0))
    denom = max(p99, 1e-6)
    mag_n = np.clip(mag / denom, 0.0, 1.0)

    return (mag_n * 255.0).astype(np.uint8)


def diff_heatmap_u8_masked(orig_bgr: np.ndarray, out_bgr: np.ndarray, mask_ignore_u8: np.ndarray) -> np.ndarray:
    """
    Like diff_heatmap_u8, but "not computed" inside mask:
    set heatmap to 0 where mask_ignore_u8 > 0.
    """
    hm = diff_heatmap_u8(orig_bgr, out_bgr)
    hm = hm.copy()
    hm[mask_ignore_u8 > 0] = 0
    return hm


# -----------------------------
# Post-processing: masked flow align + (optional) composite
# -----------------------------
def align_qwen_to_original(
    flow: MaskedOpticalFlow,
    original_resized_path: Path,
    qwen_resized_path: Path,
    out_path: Path,
    final_mode: str = "composite",  # "composite" | "aligned_only"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    - Align qwen -> original using MaskedOpticalFlow (ECC+DIS) ignoring clothing union mask
    - If final_mode == "aligned_only": final = aligned_qwen
      else: final = composite(outside mask = original, inside mask = aligned_qwen)
    Returns:
        (orig_bgr, final_bgr, mask_ignore_u8, aligned_bgr)
    """
    import cv2

    tmp_aligned = out_path.with_suffix(".tmp_aligned.png")

    info = flow.run_from_paths(
        src_path=qwen_resized_path,            # src = qwen
        tgt_path=original_resized_path,        # tgt = original
        output_path=tmp_aligned,
        debug_dir=None,
    )

    orig = cv2.imread(str(original_resized_path))
    if orig is None:
        raise FileNotFoundError(f"Could not load original resized: {original_resized_path}")

    aligned = cv2.imread(str(tmp_aligned))
    if aligned is None:
        raise FileNotFoundError(f"Could not load aligned temp output: {tmp_aligned}")

    if orig.shape[:2] != aligned.shape[:2]:
        aligned = cv2.resize(aligned, (orig.shape[1], orig.shape[0]), interpolation=cv2.INTER_LINEAR)

    mask_ignore = info["mask_ignore"]  # 255 in clothing union (dilated), 0 elsewhere

    if final_mode == "aligned_only":
        final = aligned
    else:
        mask = (mask_ignore > 0)
        final = orig.copy()
        final[mask] = aligned[mask]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(out_path), final)
    if not ok:
        raise IOError(f"cv2.imwrite failed for: {out_path}")

    try:
        tmp_aligned.unlink(missing_ok=True)
    except Exception:
        pass

    return orig, final, mask_ignore, aligned


# -----------------------------
# Main pipeline
# -----------------------------
def run_qwen_tree(
    input_root: Path,
    output_root: Path,
    garment_word: str,
    clothing_flag: str,           # upper|lower|dress
    length_flag: Optional[str],   # short|long|None
    final_mode: str = "composite",  # "composite" | "aligned_only"
    model_path: str = "Qwen/Qwen-Image-Edit-2511",
    sapiens_repo: str | Path = "Sapiens-Pytorch-Inference",
    sapiens_variant: str = "SEGMENTATION_1B",
    width: int = 704,
    height: int = 1248,
    true_cfg_scale: float = 4.0,
    num_inference_steps: int = 20,
    seed: Optional[int] = None,
    wandb_project: str = "vton_pipeline",
    wandb_run_name: Optional[str] = None,
    wandb_mode: Optional[str] = None,  # "online"|"offline"|"disabled"|None
):
    input_root = Path(input_root).resolve()
    output_root = Path(output_root).resolve()

    if not input_root.exists():
        raise FileNotFoundError(f"input_root not found: {input_root}")

    # wandb init
    cfg_dict = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "garment_word": garment_word,
        "clothing_flag": clothing_flag,
        "length_flag": length_flag,
        "final_mode": final_mode,
        "model_path": model_path,
        "sapiens_repo": str(sapiens_repo),
        "sapiens_variant": sapiens_variant,
        "width": width,
        "height": height,
        "true_cfg_scale": true_cfg_scale,
        "num_inference_steps": num_inference_steps,
        "seed": seed,
        "negative_prompt": NEGATIVE_PROMPT,
        "edit_mode": "single_image_no_reference",
    }

    if wandb_mode is not None:
        wandb.init(project=wandb_project, name=wandb_run_name, mode=wandb_mode, config=cfg_dict)
    else:
        wandb.init(project=wandb_project, name=wandb_run_name, config=cfg_dict)

    # init once
    pipe = load_qwen_pipe(model_path=model_path)

    # IMPORTANT: for dress we want union mask to include:
    # upper clothing + lower clothing + arms + legs.
    # Your MaskedOpticalFlowConfig (as pasted earlier) already does that when clothing_flag == "dress".
    # We enforce it here by passing clothing_flag="dress" and length_flag=None.
    if clothing_flag == "dress":
        length_flag = None

    flow_cfg = MaskedOpticalFlowConfig(
        target_h=height,
        target_w=width,
        sapiens_repo=sapiens_repo,
        sapiens_variant=sapiens_variant,
        clothing_flag=clothing_flag,
        length_flag=length_flag,
    )
    flow = MaskedOpticalFlow(flow_cfg)

    in_paths = sorted([p for p in input_root.rglob("*") if p.is_file() and is_image_file(p)])

    # temp dir for intermediates
    tmp_dir = output_root / ".tmp_qwen"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    done = 0
    copied_flat = 0
    failed = 0

    import cv2

    for idx, p in enumerate(in_paths, start=1):
        rel = p.relative_to(input_root)
        out_path = output_root / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)

        failed_path = output_root / "failed" / rel
        failed_path.parent.mkdir(parents=True, exist_ok=True)

        # "flat" => copy only (not processed)
        if is_flat(p):
            try:
                shutil.copy2(p, out_path)
                copied_flat += 1
                wandb.log({"path": rel.as_posix(), "status": "flat_copied"}, step=idx)
            except Exception as e:
                failed += 1
                try:
                    shutil.copy2(p, failed_path)
                except Exception:
                    pass
                wandb.log({"path": rel.as_posix(), "status": "flat_copy_failed", "error": str(e)}, step=idx)
            continue

        # normal => qwen + flow + (optional) composite + log images + heatmaps
        try:
            # Load original (PIL for qwen)
            person_img = Image.open(p).convert("RGB")

            # Qwen output (already at width/height because we pass them)
            qwen_img = qwen_edit_one(
                pipe=pipe,
                person_img=person_img,
                garment_word=garment_word,
                true_cfg_scale=true_cfg_scale,
                num_inference_steps=num_inference_steps,
                width=width,
                height=height,
                seed=seed,
            )

            # Prepare resized original (BGR) to target size for fair compare & flow
            orig_bgr = cv2.imread(str(p))
            if orig_bgr is None:
                raise FileNotFoundError(f"cv2 could not read: {p}")

            interp = cv2.INTER_AREA if (orig_bgr.shape[0] > height or orig_bgr.shape[1] > width) else cv2.INTER_LINEAR
            orig_resized = cv2.resize(orig_bgr, (width, height), interpolation=interp)

            # Write intermediates as temp files so we can run your flow class unchanged (it reads from disk)
            key = rel.as_posix().replace("/", "__")
            orig_tmp_path = tmp_dir / f"{key}.orig_resized.png"
            qwen_tmp_path = tmp_dir / f"{key}.qwen.png"

            cv2.imwrite(str(orig_tmp_path), orig_resized)
            qwen_img.save(qwen_tmp_path)

            # Ensure output extension is an image; default keep original suffix, else .png
            out_save_path = out_path
            if out_save_path.suffix.lower() not in (".png", ".jpg", ".jpeg"):
                out_save_path = out_save_path.with_suffix(".png")

            # Align (and optionally composite), then save final
            orig_bgr2, final_bgr, mask_ignore, aligned_bgr = align_qwen_to_original(
                flow=flow,
                original_resized_path=orig_tmp_path,
                qwen_resized_path=qwen_tmp_path,
                out_path=out_save_path,
                final_mode=final_mode,
            )

            # Heatmaps (NOT saved, only logged): original vs FINAL
            hm_full = diff_heatmap_u8(orig_bgr2, final_bgr)
            hm_masked = diff_heatmap_u8_masked(orig_bgr2, final_bgr, mask_ignore)

            # Log EVERYTHING for this image
            wandb.log(
                {
                    "path": rel.as_posix(),
                    "status": "ok",
                    "final_mode": final_mode,
                    "original": bgr_to_wandb_image(orig_bgr2, caption=rel.as_posix()),
                    "qwen": wandb.Image(np.asarray(qwen_img), caption=rel.as_posix()),
                    "aligned_qwen": bgr_to_wandb_image(aligned_bgr, caption=f"{rel.as_posix()} | aligned"),
                    "final": bgr_to_wandb_image(final_bgr, caption=f"{rel.as_posix()} | final"),
                    "heatmap_full": gray_u8_to_wandb_image(hm_full, caption="orig vs final"),
                    "heatmap_masked": gray_u8_to_wandb_image(hm_masked, caption="orig vs final (mask ignored)"),
                    "mask_ignore": gray_u8_to_wandb_image(mask_ignore, caption="union mask (dilated)"),
                },
                step=idx,
            )

            done += 1

        except Exception as e:
            failed += 1
            try:
                shutil.copy2(p, failed_path)
            except Exception:
                pass

            wandb.log({"path": rel.as_posix(), "status": "failed", "error": str(e)}, step=idx)
            print(f"[FAILED] {p} -> {e}")

    # cleanup temps (comment out if you want to inspect intermediates)
    try:
        shutil.rmtree(tmp_dir)
    except Exception:
        pass

    print("----- DONE -----")
    print(f"Total images found: {len(in_paths)}")
    print(f"Processed OK:       {done}")
    print(f"Copied flat:        {copied_flat}")
    print(f"Failed copied:      {failed}")
    print(f"Output root:        {output_root}")

    wandb.finish()


def _normalize_length_flag(x: Optional[str]) -> Optional[str]:
    if x is None:
        return None
    x = x.strip().lower()
    if x in ("none", "null", ""):
        return None
    if x not in ("short", "long"):
        raise ValueError("length_flag must be short|long|none")
    return x


def main():
    ap = argparse.ArgumentParser(
        "Recursive Qwen edit (NO reference clothing image) + masked optical flow (union clothing mask), "
        "preserving folder structure, with W&B logging + 2 heatmaps."
    )
    ap.add_argument("--input_root", type=Path, required=True)
    ap.add_argument("--output_root", type=Path, required=True)

    ap.add_argument("--garment_word", type=str, required=True, help='Inserted into prompt, e.g. "shirt", "jacket".')

    ap.add_argument("--clothing_flag", type=str, required=True, choices=["upper", "lower", "dress"])
    ap.add_argument("--length_flag", type=str, default="none", help="short|long|none (use none for dress).")

    ap.add_argument(
        "--final_mode",
        type=str,
        default="composite",
        choices=["composite", "aligned_only"],
        help=(
            "composite: outside mask=original, inside mask=aligned qwen. "
            "aligned_only: output is the aligned qwen image only (no pixel replacement from original)."
        ),
    )

    ap.add_argument("--model_path", type=str, default="Qwen/Qwen-Image-Edit-2511")
    ap.add_argument("--sapiens_repo", type=str, default="Sapiens-Pytorch-Inference")
    ap.add_argument("--sapiens_variant", type=str, default="SEGMENTATION_1B")

    ap.add_argument("--width", type=int, default=704)
    ap.add_argument("--height", type=int, default=1248)
    ap.add_argument("--true_cfg_scale", type=float, default=4.0)
    ap.add_argument("--num_inference_steps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=None)

    ap.add_argument("--wandb_project", type=str, default="vton_pipeline")
    ap.add_argument("--wandb_run_name", type=str, default=None)
    ap.add_argument("--wandb_mode", type=str, default=None, help="online|offline|disabled")

    args = ap.parse_args()

    clothing_flag = args.clothing_flag
    length_flag = _normalize_length_flag(args.length_flag)

    # If dress: force length_flag=None; and MaskedOpticalFlowConfig will include upper+lower+arms+legs.
    if clothing_flag == "dress":
        length_flag = None

    run_qwen_tree(
        input_root=args.input_root,
        output_root=args.output_root,
        garment_word=args.garment_word,
        clothing_flag=clothing_flag,
        length_flag=length_flag,
        final_mode=args.final_mode,
        model_path=args.model_path,
        sapiens_repo=args.sapiens_repo,
        sapiens_variant=args.sapiens_variant,
        width=args.width,
        height=args.height,
        true_cfg_scale=args.true_cfg_scale,
        num_inference_steps=args.num_inference_steps,
        seed=args.seed,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        wandb_mode=args.wandb_mode,
    )


if __name__ == "__main__":
    main()
