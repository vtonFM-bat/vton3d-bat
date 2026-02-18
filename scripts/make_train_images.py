#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List
import shutil

import numpy as np
from PIL import Image

import torch
from transformers import Sam3Processor, Sam3Model


# -----------------------------
# SAM3 Human Segmentation
# -----------------------------
@dataclass
class Sam3HumanSegConfig:
    model_id: str = "facebook/sam3"
    prompt: str = "human"
    threshold: float = 0.5
    mask_threshold: float = 0.5
    pick: str = "union"  # "union", "largest", "best_score"
    device: Optional[str] = None


class Sam3HumanSegmenter:
    def __init__(self, cfg: Sam3HumanSegConfig):
        self.cfg = cfg

        if cfg.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = cfg.device

        if cfg.pick not in ("union", "largest", "best_score"):
            raise ValueError(f"pick must be union|largest|best_score, got: {cfg.pick}")

        self.model = Sam3Model.from_pretrained(cfg.model_id).to(self.device)
        self.processor = Sam3Processor.from_pretrained(cfg.model_id)
        self.model.eval()

    @torch.no_grad()
    def segment_human_mask(self, image: Image.Image) -> Optional[np.ndarray]:
        """
        Returns:
          human_mask_bool(H, W) or None
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
            return None

        masks_np = masks.detach().cpu().numpy().astype(bool)  # (N, H, W)

        if self.cfg.pick == "union":
            human_mask = np.any(masks_np, axis=0)

        elif self.cfg.pick == "largest":
            areas = masks_np.reshape(masks_np.shape[0], -1).sum(axis=1)
            idx = int(np.argmax(areas))
            human_mask = masks_np[idx]

        else:  # "best_score"
            if scores is None or len(scores) == 0:
                human_mask = np.any(masks_np, axis=0)
            else:
                scores_np = scores.detach().cpu().numpy().astype(float)
                idx = int(np.argmax(scores_np))
                human_mask = masks_np[idx]

        return human_mask


# -----------------------------
# Image utils
# -----------------------------
def load_background_paths(bg_root: Path, exts=(".png", ".jpg", ".jpeg")) -> List[Path]:
    paths = []
    for p in bg_root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            paths.append(p)
    return sorted(paths)


def fit_background_to_size(bg: Image.Image, size: Tuple[int, int]) -> Image.Image:
    """
    Fit bg to (W,H) by center-cropping after resize (cover).
    """
    target_w, target_h = size
    bg = bg.convert("RGB")

    w, h = bg.size
    if w == 0 or h == 0:
        raise ValueError("Background image has invalid size.")

    scale = max(target_w / w, target_h / h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    bg_resized = bg.resize((new_w, new_h), Image.LANCZOS)

    left = (new_w - target_w) // 2
    top = (new_h - target_h) // 2
    right = left + target_w
    bottom = top + target_h

    return bg_resized.crop((left, top, right, bottom))


def composite_human_on_random_bg(
    image: Image.Image,
    human_mask: np.ndarray,
    bg_img: Image.Image,
    feather_px: int = 2,
) -> Image.Image:
    """
    Composites human (foreground) onto bg using mask.
    Returns RGB image (same size as input).
    """
    img_rgb = image.convert("RGB")
    w, h = img_rgb.size

    # background to same size
    bg_fit = fit_background_to_size(bg_img, (w, h))

    # mask -> L image
    mask_u8 = (human_mask.astype(np.uint8) * 255)
    mask = Image.fromarray(mask_u8, mode="L")

    # optional small feather to avoid harsh edges
    if feather_px and feather_px > 0:
        # cheap feather via resize trick (fast, no extra deps)
        # shrink then grow to soften
        small_w = max(1, w // (feather_px * 2))
        small_h = max(1, h // (feather_px * 2))
        mask = mask.resize((small_w, small_h), Image.BILINEAR).resize((w, h), Image.BILINEAR)

    out = Image.composite(img_rgb, bg_fit, mask)
    return out


def resize_exact(img: Image.Image, target_h: int, target_w: int) -> Image.Image:
    return img.resize((target_w, target_h), Image.LANCZOS)


# -----------------------------
# Main processing
# -----------------------------
def is_image_file(p: Path) -> bool:
    return p.suffix.lower() in (".png", ".jpg", ".jpeg")


def should_skip_flat(p: Path) -> bool:
    return "flat" in p.name.lower()


def process_tree(
    input_root: Path,
    bg_root: Path,
    output_root: Path,
    target_h: int,
    target_w: int,
    cfg: Sam3HumanSegConfig,
    seed: Optional[int] = None,
):
    if seed is not None:
        random.seed(seed)

    input_root = input_root.resolve()
    bg_root = bg_root.resolve()
    output_root = output_root.resolve()

    if not input_root.exists():
        raise FileNotFoundError(f"Input root not found: {input_root}")
    if not bg_root.exists():
        raise FileNotFoundError(f"Background root not found: {bg_root}")

    bg_paths = load_background_paths(bg_root)
    if len(bg_paths) == 0:
        raise FileNotFoundError(f"No background images found in: {bg_root}")

    segmenter = Sam3HumanSegmenter(cfg)

    in_paths = sorted([p for p in input_root.rglob("*") if p.is_file() and is_image_file(p)])

    total = len(in_paths)
    done = 0
    skipped_flat = 0
    skipped_no_human = 0
    failed = 0

    for p in in_paths:
        rel = p.relative_to(input_root)

        # Standard-Ziel (normaler Output-Baum)
        out_path = output_root / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Failed-Ziel (separater Baum)
        failed_path = output_root / "failed" / rel
        failed_path.parent.mkdir(parents=True, exist_ok=True)

        # 1) "flat" -> nur kopieren, nicht bearbeiten
        if should_skip_flat(p):
            try:
                shutil.copy2(p, out_path)
                skipped_flat += 1
            except Exception as e:
                failed += 1
                print(f"[ERROR][copy flat] {p} -> {e}")
            continue

        # 2) Alle anderen -> bearbeiten
        try:
            img = Image.open(p).convert("RGB")

            human_mask = segmenter.segment_human_mask(img)
            if human_mask is None:
                # kein Mensch -> in failed kopieren
                try:
                    shutil.copy2(p, failed_path)
                except Exception as e:
                    print(f"[ERROR][copy failed no-human] {p} -> {e}")
                skipped_no_human += 1
                continue

            bg_path = random.choice(bg_paths)
            bg_img = Image.open(bg_path).convert("RGB")

            composed = composite_human_on_random_bg(img, human_mask, bg_img, feather_px=2)
            composed = resize_exact(composed, target_h=target_h, target_w=target_w)

            # Speichern im normalen Output-Baum
            suffix = out_path.suffix.lower()
            if suffix in (".jpg", ".jpeg"):
                composed.save(out_path, format="JPEG", quality=95)
            else:
                composed.save(out_path, format="PNG")

            done += 1

        except Exception as e:
            # Exception -> Original in failed kopieren
            try:
                shutil.copy2(p, failed_path)
            except Exception as e2:
                print(f"[ERROR][copy failed exception] {p} -> {e2}")

            failed += 1
            print(f"[ERROR][process] {p} -> {e}")

    print("----- DONE -----")
    print(f"Input images found:        {total}")
    print(f"Processed successfully:    {done}")
    print(f"Skipped (name has 'flat'): {skipped_flat}")
    print(f"Skipped (no human found):  {skipped_no_human}")
    print(f"Failed:                   {failed}")
    print(f"Output root:              {output_root}")


def parse_args():
    ap = argparse.ArgumentParser(
        description="Recursively segment human with SAM3, replace background with random bg, resize, and write outputs preserving folder structure."
    )
    ap.add_argument("--input_root", type=Path, required=True, help="Root folder with images (recursive).")
    ap.add_argument("--bg_root", type=Path, required=True, help="Folder that contains random background images (recursive).")
    ap.add_argument("--output_root", type=Path, required=True, help="Output root folder (same structure as input).")

    ap.add_argument("--target_h", type=int, default=1248, help="Target height (default: 1248).")
    ap.add_argument("--target_w", type=int, default=704, help="Target width (default: 704).")

    ap.add_argument("--model_id", type=str, default="facebook/sam3")
    ap.add_argument("--prompt", type=str, default="human")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--mask_threshold", type=float, default=0.5)
    ap.add_argument("--pick", type=str, default="union", choices=["union", "largest", "best_score"])
    ap.add_argument("--device", type=str, default=None, help="cuda|cpu (default: auto).")

    ap.add_argument("--seed", type=int, default=None, help="Optional random seed for background choice.")

    return ap.parse_args()


def main():
    args = parse_args()

    cfg = Sam3HumanSegConfig(
        model_id=args.model_id,
        prompt=args.prompt,
        threshold=args.threshold,
        mask_threshold=args.mask_threshold,
        pick=args.pick,
        device=args.device,
    )

    process_tree(
        input_root=args.input_root,
        bg_root=args.bg_root,
        output_root=args.output_root,
        target_h=args.target_h,
        target_w=args.target_w,
        cfg=cfg,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
