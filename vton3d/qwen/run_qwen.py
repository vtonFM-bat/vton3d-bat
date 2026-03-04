import argparse
from pathlib import Path
import gc
import sys
from PIL import Image
import torch
from diffusers import QwenImageEditPlusPipeline
import wandb
import cv2
import matplotlib.cm as cm
import numpy as np
import bisect
from safetensors.torch import load_file, save_file
import tempfile

from vton3d.utils.qwen_eval import (
    qwen_eval_masked,
    qwen_fashionclip_similarity_masked_clothing,
    qwen_arcface_similarity_input_vs_output,
    qwen_fashionclip_similarity_neighbor_masked_clothing,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SAPIENS_REPO = REPO_ROOT / "Sapiens-Pytorch-Inference"
sys.path.insert(0, str(SAPIENS_REPO))

from sapiens_inference.segmentation import SapiensSegmentation, SapiensSegmentationType


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch Qwen Image Edit pipeline.")

    parser.add_argument("--model_path", type=str, default="Qwen/Qwen-Image-Edit-2511")
    parser.add_argument("--source_dir", type=str, required=True)
    parser.add_argument("--clothing_image", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument(
        "--prompt",
        type=str,
        default=(
            "Replace the person's original clothing with the clothing image. "
            "Do not invent new features. Keep the person identical: pose, body, "
            "face, hair, and background must remain unchanged."
        ),
    )
    parser.add_argument("--negative_prompt", type=str, default="change pose")

    parser.add_argument("--true_cfg_scale", type=float, default=4.0)
    parser.add_argument("--num_inference_steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--extensions", type=str, nargs="+", default=[".jpg", ".jpeg", ".png"])

    parser.add_argument("--use_lora", type=str, default="", help="Path to LoRA .safetensors (empty => disable).")
    parser.add_argument("--lora_scale", type=float, default=1.0, help="LoRA scale (only if use_lora is set).")

    return parser.parse_args()


def infer_eval_flag_from_clothing_path(clothing_path: Path) -> str:
    parts = [p.lower() for p in clothing_path.parts]

    if "dress" in parts:
        return "dress"

    if "lower" in parts and "upper" in parts:
        raise ValueError(f"Ambiguous clothing path contains both 'upper' and 'lower': {clothing_path}")

    if "lower" in parts:
        return "lower"
    if "upper" in parts:
        return "upper"

    raise ValueError(f"Could not infer eval_flag from clothing path: {clothing_path}")


def infer_length_flag_from_clothing_path(clothing_path: Path) -> str:
    parts = [p.lower() for p in clothing_path.parts]

    if "long" in parts and "short" in parts:
        raise ValueError(f"Ambiguous clothing path contains both 'long' and 'short': {clothing_path}")

    if "long" in parts:
        return "long"
    if "short" in parts:
        return "short"

    raise ValueError(f"Could not infer length_flag from clothing path: {clothing_path}")


def load_pipeline(model_path: str, use_lora: str = "", lora_scale: float = 1.0) -> QwenImageEditPlusPipeline:
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
    )
    pipe.set_progress_bar_config(disable=None)
    pipe.enable_model_cpu_offload()

    use_lora = (use_lora or "").strip()
    if use_lora:
        lora_path = Path(use_lora).expanduser().resolve()
        adapter_name = "lora"

        if not lora_path.exists():
            raise FileNotFoundError(f"LoRA path not found: {lora_path}")

        if lora_path.is_file() and lora_path.suffix == ".safetensors":
            sd = load_file(str(lora_path))

            # remove extra training heads
            drop_prefixes = ("vggt_loss.",)
            keys_before = list(sd.keys())
            sd_filtered = {k: v for k, v in sd.items() if not k.startswith(drop_prefixes)}
            dropped = [k for k in keys_before if k not in sd_filtered]

            if len(sd_filtered) == 0:
                raise ValueError(f"After filtering, no keys left in LoRA file. Dropped={len(dropped)}")

            if dropped:
                print(f"[qwen] Filtering LoRA safetensors: dropped {len(dropped)} keys (e.g. {dropped[:4]})")

            with tempfile.TemporaryDirectory() as td:
                td = Path(td)
                filtered_path = td / lora_path.name
                save_file(sd_filtered, str(filtered_path))

                pipe.load_lora_weights(
                    str(filtered_path.parent),
                    weight_name=filtered_path.name,
                    adapter_name=adapter_name,
                )
        else:
            pipe.load_lora_weights(str(lora_path), adapter_name=adapter_name)

        try:
            pipe.set_adapters(adapter_name, adapter_weights=float(lora_scale))
        except Exception:
            pass

        try:
            pipe.fuse_lora(lora_scale=float(lora_scale))
        except Exception:
            try:
                pipe.fuse_lora()
            except Exception:
                pass

        print(f"[qwen] LoRA enabled: {use_lora} (scale={lora_scale})")
    else:
        print("[qwen] LoRA disabled (base model only)")

    return pipe


import re


def _extract_frame_number(p: Path) -> int:
    m = re.search(r"_([0-9]+)\.[^.]+$", p.name)
    if not m:
        return 10**18
    return int(m.group(1))


def get_image_files(source_dir: Path, extensions: list[str]) -> list[Path]:
    exts = {e.lower() for e in extensions}
    files = [p for p in source_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
    return sorted(files, key=_extract_frame_number)


def concat_refs_side_by_side(img_left: Image.Image, img_right: Image.Image) -> Image.Image:
    img_left = img_left.convert("RGB")
    img_right = img_right.convert("RGB")

    h = max(img_left.height, img_right.height)
    w = img_left.width + img_right.width
    canvas = Image.new("RGB", (w, h))
    canvas.paste(img_left, (0, 0))
    canvas.paste(img_right, (img_left.width, 0))
    return canvas


def clear_gpu_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def align_and_overwrite_with_optical_flow(
    aligner,
    src_path: Path,
    tgt_path: Path,
    composite_original_outside_mask: bool,
):
    result = aligner.run_from_paths(
        src_path=src_path,
        tgt_path=tgt_path,
        output_path=src_path,
        debug_dir=None,
    )

    aligned_bgr = cv2.imread(str(src_path))
    if aligned_bgr is None:
        raise RuntimeError(f"Could not read aligned image after write: {src_path}")

    mask = result["mask_ignore"]

    real_bgr = cv2.imread(str(tgt_path))
    if real_bgr is None:
        raise RuntimeError(f"Could not read real image: {tgt_path}")

    if composite_original_outside_mask:
        inside = (mask > 0)
        comp_bgr = real_bgr.copy()
        comp_bgr[inside] = aligned_bgr[inside]
        ok = cv2.imwrite(str(src_path), comp_bgr)
        if not ok:
            raise IOError(f"cv2.imwrite failed for composite: {src_path}")
        aligned_bgr = comp_bgr

    aligned_rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB)

    real = real_bgr.astype(np.float32) / 255.0
    aligned = aligned_bgr.astype(np.float32) / 255.0

    mask_ignore = mask.astype(bool)
    mask_include = ~mask_ignore

    diff = (real - aligned)
    abs_diff = np.abs(diff)
    heatmap_gray = abs_diff.mean(axis=2)
    heatmap_gray[mask_ignore] = 0.0

    norm = heatmap_gray / (heatmap_gray.max() + 1e-8)
    heatmap_rgb = (cm.get_cmap("Reds")(norm)[..., :3] * 255).astype(np.uint8)

    diff2 = diff**2
    diff2_masked = diff2[mask_include]
    mse = float(diff2_masked.mean()) if diff2_masked.size > 0 else float("nan")
    psnr = float(10.0 * np.log10(1.0 / (mse + 1e-12))) if np.isfinite(mse) else float("nan")

    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

    flow = result["flow"]
    mag = np.linalg.norm(flow, axis=2)
    mean_mag = float(mag.mean())

    mag_img = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    mag_rgb = cv2.cvtColor(mag_img, cv2.COLOR_GRAY2RGB)

    return {
        "aligned_rgb": aligned_rgb,
        "mask_rgb": mask_rgb,
        "flow_mag_rgb": mag_rgb,
        "heatmap_rgb": heatmap_rgb,
        "mse": mse,
        "psnr": psnr,
        "mean_flow_magnitude": mean_mag,
    }


def run_mof_and_get_aligned_pil(
    aligner,
    out_path: Path,
    real_path: Path,
    composite_original_outside_mask: bool,
):
    oflow = align_and_overwrite_with_optical_flow(
        aligner=aligner,
        src_path=out_path,
        tgt_path=real_path,
        composite_original_outside_mask=composite_original_outside_mask,
    )

    wandb.log({
        "opticalflow/aligned": wandb.Image(oflow["aligned_rgb"], caption=out_path.name),
        "opticalflow/mask": wandb.Image(oflow["mask_rgb"], caption=f"{out_path.stem}_mask"),
        "opticalflow/flow_map": wandb.Image(oflow["flow_mag_rgb"], caption=f"{out_path.stem}_flow_mag"),
        "opticalflow/mean_flow_magnitude": oflow["mean_flow_magnitude"],
        "opticalflow/heatmap_diff": wandb.Image(oflow["heatmap_rgb"], caption=f"{out_path.stem}_heatmap_diff"),
        "opticalflow/mse_post_align": oflow["mse"],
        "opticalflow/psnr_post_align": oflow["psnr"],
    })

    return Image.open(out_path).convert("RGB")


def run_qwen_from_config_dict(qwen_cfg: dict):
    wandb.define_metric("qwen/*", step_metric="qwen/image_index")

    model_path = qwen_cfg.get("model_path", "Qwen/Qwen-Image-Edit-2511")
    source_dir = Path(qwen_cfg["source_dir"])
    clothing_path = Path(qwen_cfg["clothing_image"])
    output_dir = Path(qwen_cfg["output_dir"])

    default_prompt = qwen_cfg.get("prompt", "")
    default_negative_prompt = qwen_cfg.get("negative_prompt", "change pose")

    true_cfg_scale = float(qwen_cfg.get("true_cfg_scale", 4.0))
    num_inference_steps = int(qwen_cfg.get("num_inference_steps", 20))
    seed = int(qwen_cfg.get("seed", 0))
    extensions = qwen_cfg.get("extensions", [".jpg", ".jpeg", ".png"])

    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")
    if not clothing_path.is_file():
        raise FileNotFoundError(f"Clothing image not found: {clothing_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    clothing_image = Image.open(clothing_path).convert("RGB")
    image_files = get_image_files(source_dir, extensions)
    n = len(image_files)
    if n == 0:
        raise RuntimeError(f"No images found in {source_dir}.")

    # frames for neighbor-eval logging only
    frame_nums = [_extract_frame_number(p) for p in image_files]
    sorted_frames = sorted(frame_nums)
    path_by_frame = {fn: p for fn, p in zip(frame_nums, image_files)}

    print(f"[frames] count={n} first={sorted_frames[:5]} last={sorted_frames[-5:]}")

    use_lora = qwen_cfg.get("use_lora", "")
    lora_scale = float(qwen_cfg.get("lora_scale", 1.0))
    pipeline = load_pipeline(model_path, use_lora=use_lora, lora_scale=lora_scale)

    wandb.log({"qwen/clothing_image": wandb.Image(clothing_image, caption=clothing_path.name)})

    estimator = SapiensSegmentation(
        SapiensSegmentationType.SEGMENTATION_1B,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        dtype=torch.float16,
    )

    eval_flag = infer_eval_flag_from_clothing_path(clothing_path)
    length_flag = None if eval_flag == "dress" else infer_length_flag_from_clothing_path(clothing_path)
    print(f"[qwen] inferred eval_flag='{eval_flag}', length_flag='{length_flag}' from clothing_image='{clothing_path}'")

    prompts_map = qwen_cfg.get("prompts", {}) or {}
    negative_prompts_map = qwen_cfg.get("negative_prompts", {}) or {}

    base_prompt = prompts_map.get(eval_flag, default_prompt)
    negative_prompt = negative_prompts_map.get(eval_flag, default_negative_prompt)

    print(f"Prompt: {base_prompt}")
    print(f"negative Prompt: {negative_prompt}")

    def tweak_base_prompt_for_n1_non_front(p: str) -> str:
        needle = "Keep background, pose, face, hair, skin, body shape, all other clothes of the person,"
        repl = "Keep background, pose, face, hair, skin, body shape, all other clothes of the person in image 1,"
        return p.replace(needle, repl, 1) if needle in p else p

    PREFIX_SINGLE_REF = (
        "The person in image 1 wears the exact garment from image 2, "
        "preserving image 1 entirely, with the garment naturally rotated "
        "and pattern-aligned according to image 3. "
    )
    PREFIX_BOTH_REFS = (
        "The person in image 1 wears the exact garment from image 2, "
        "preserving image 1 entirely, with the garment naturally rotated "
        "and pattern-aligned according to image 3 and pattern-aligned according to image 4. "
    )

    def build_prompt(ref_kind: str, use_n_1: bool) -> str:
        if not use_n_1:
            return base_prompt
        if ref_kind == "front":
            return base_prompt
        base_n1 = tweak_base_prompt_for_n1_non_front(base_prompt)
        if ref_kind == "single":
            return PREFIX_SINGLE_REF + base_n1
        if ref_kind == "both":
            return PREFIX_BOTH_REFS + base_n1
        return base_n1

    # optional optical flow
    mof_cfg_dict = qwen_cfg.get("_masked_optical_flow", None)
    do_integrated_mof = mof_cfg_dict is not None
    aligner = None
    composite_original_outside_mask = False

    if do_integrated_mof:
        from vton3d.utils.masked_optical_flow import MaskedOpticalFlow, MaskedOpticalFlowConfig

        composite_original_outside_mask = bool(mof_cfg_dict.get("composite_original_outside_mask", False))
        mof_cfg = MaskedOpticalFlowConfig(
            target_h=int(mof_cfg_dict.get("target_h", 1248)),
            target_w=int(mof_cfg_dict.get("target_w", 704)),
            sapiens_repo=mof_cfg_dict.get("sapiens_repo", "Sapiens-Pytorch-Inference"),
            sapiens_variant=mof_cfg_dict.get("sapiens_variant", "SEGMENTATION_1B"),
            dilate_px=int(mof_cfg_dict.get("dilate_px", 10)),
            feather_sigma=float(mof_cfg_dict.get("feather_sigma", 7.0)),
            ecc_n_iter=int(mof_cfg_dict.get("ecc_n_iter", 400)),
            ecc_eps=float(mof_cfg_dict.get("ecc_eps", 1e-7)),
            flag_source_path=qwen_cfg.get("clothing_image", None),
        )
        aligner = MaskedOpticalFlow(mof_cfg)

    def log_neighbor_fashionclip(curr_img_path: Path, curr_out_path: Path):
        f = _extract_frame_number(curr_img_path)
        pos = bisect.bisect_left(sorted_frames, f)

        if pos <= 0:
            wandb.log({f"qwen/fc_neighbor_prev_{eval_flag}_{length_flag}": float("nan")})
        else:
            prev_f = sorted_frames[pos - 1]
            prev_img = path_by_frame[prev_f]
            prev_out_path = output_dir / f"{prev_img.stem}.png"
            if prev_out_path.exists():
                sim, m_prev, m_curr = qwen_fashionclip_similarity_neighbor_masked_clothing(
                    person_img_a_path=str(prev_out_path),
                    person_img_b_path=str(curr_out_path),
                    flag=eval_flag,
                    estimator=estimator,
                    clip_device="cuda",
                    return_masked_rgb=True,
                )
                wandb.log({
                    f"qwen/fc_neighbor_prev_{eval_flag}_{length_flag}": sim,
                    f"qwen/masked_neighbor_prev_{eval_flag}": wandb.Image(m_prev, caption=f"{prev_out_path.stem}_masked_prev"),
                    f"qwen/masked_neighbor_curr_{eval_flag}": wandb.Image(m_curr, caption=f"{curr_out_path.stem}_masked_curr"),
                })
            else:
                wandb.log({f"qwen/fc_neighbor_prev_{eval_flag}_{length_flag}": float("nan")})

    img_count = 0

    def save_and_align(img_path: Path, out_img_raw: Image.Image) -> tuple[Path, Image.Image]:
        out_path = output_dir / f"{img_path.stem}.png"
        out_img_raw.save(out_path)

        out_img_for_ref = out_img_raw
        if aligner is not None:
            try:
                out_img_for_ref = run_mof_and_get_aligned_pil(
                    aligner=aligner,
                    out_path=out_path,
                    real_path=Path(img_path),
                    composite_original_outside_mask=composite_original_outside_mask,
                )
            except Exception as e:
                print(f"[WARN] optical flow failed for {out_path.name}: {e}")

        return out_path, out_img_for_ref

    def post_eval_and_log(img_path: Path, out_path: Path, out_img_raw: Image.Image, use_n1_value: int):
        nonlocal img_count

        log_neighbor_fashionclip(curr_img_path=img_path, curr_out_path=out_path)

        mse_value, psnr_value, heatmap = qwen_eval_masked(
            img1_path=str(img_path),
            img2_path=str(out_path),
            flag=eval_flag,
            length_flag=length_flag,
            estimator=estimator,
        )

        face_sim, _, _ = qwen_arcface_similarity_input_vs_output(
            img1_path=str(img_path),
            img2_path=str(out_path),
            device="cuda",
            det_size=(640, 640),
            return_faces_rgb=True,
        )

        fc_sim, masked_rgb = qwen_fashionclip_similarity_masked_clothing(
            person_img_path=str(out_path),
            clothing_ref_path=str(clothing_path),
            flag=eval_flag,
            estimator=estimator,
            clip_device="cuda",
            return_masked_rgb=True,
        )

        img_count += 1
        wandb.log({
            "qwen/output_image": wandb.Image(out_img_raw, caption=out_path.name),
            "qwen/image_index": img_count,
            "qwen/use_n_1": use_n1_value,
            f"qwen/mse_non_clothed_area_{eval_flag}_{length_flag}": mse_value,
            f"qwen/psnr_non_clothed_area_{eval_flag}_{length_flag}": psnr_value,
            f"qwen/heatmap_non_clothed_area_{eval_flag}_{length_flag}": wandb.Image(
                heatmap, caption=f"{img_path.stem}_heatmap_{eval_flag}_{length_flag}"
            ),
            f"qwen/fashionclip_sim_input_{eval_flag}_clothing": fc_sim,
            f"qwen/masked_input_clothing_{eval_flag}": wandb.Image(masked_rgb, caption=f"{img_path.stem}_masked_input_{eval_flag}"),
            "qwen/face_sim_input_vs_output": face_sim,
        })

    def log_inputs(person_image: Image.Image, img_path: Path, ref_kind: str, ref_a: Image.Image | None = None, ref_b: Image.Image | None = None):
        payload = {
            "qwen/input_image": wandb.Image(person_image, caption=img_path.name),
            "qwen/clothing_image": wandb.Image(clothing_image, caption=clothing_path.name),
            "qwen/ref_kind": ref_kind,
        }
        if ref_a is not None:
            payload["qwen/ref_a"] = wandb.Image(ref_a, caption=f"{img_path.stem}_ref_a")
        if ref_b is not None:
            payload["qwen/ref_b"] = wandb.Image(ref_b, caption=f"{img_path.stem}_ref_b")
        wandb.log(payload)

    use_n_1 = bool(qwen_cfg.get("use_n_1", False))

    if not use_n_1:
        for img_path in image_files:
            person_image = Image.open(img_path).convert("RGB")
            gen = torch.Generator(device="cpu").manual_seed(seed)

            with torch.inference_mode():
                out = pipeline(
                    image=[person_image, clothing_image],
                    prompt=build_prompt("front", use_n_1),
                    generator=gen,
                    true_cfg_scale=true_cfg_scale,
                    negative_prompt=negative_prompt,
                    num_inference_steps=num_inference_steps,
                    width=704,
                    height=1248,
                )

            out_img_raw = out.images[0]
            out_path, _ = save_and_align(img_path, out_img_raw)
            wandb.log({"qwen/input_image": wandb.Image(person_image, caption=img_path.name), "qwen/use_n_1": 0})
            post_eval_and_log(img_path, out_path, out_img_raw, use_n1_value=0)
            clear_gpu_cache()
        return

    front_idx = 0
    back_idx = (n - 1) // 2  # for 30 frames = 14

    print(f"[qwen] seeds: front_idx={front_idx}, back_idx={back_idx}, n={n}")

    pred_by_idx: dict[int, Image.Image] = {}
    generated: set[int] = set()

    def gen_solo(idx: int, tag: str) -> Image.Image:
        img_path = image_files[idx]
        person = Image.open(img_path).convert("RGB")
        log_inputs(person, img_path, ref_kind=f"solo_{tag}")

        gen = torch.Generator(device="cpu").manual_seed(seed)
        with torch.inference_mode():
            out = pipeline(
                image=[person, clothing_image],
                prompt=build_prompt("front", use_n_1),
                generator=gen,
                true_cfg_scale=true_cfg_scale,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                width=704,
                height=1248,
            )

        out_raw = out.images[0]
        out_path, out_for_ref = save_and_align(img_path, out_raw)
        pred_by_idx[idx] = out_for_ref
        generated.add(idx)

        post_eval_and_log(img_path, out_path, out_raw, use_n1_value=1)
        clear_gpu_cache()
        return out_for_ref

    def gen_single(idx: int, ref_img: Image.Image, tag: str) -> Image.Image:
        img_path = image_files[idx]
        person = Image.open(img_path).convert("RGB")
        log_inputs(person, img_path, ref_kind=f"single_{tag}", ref_a=ref_img)

        gen = torch.Generator(device="cpu").manual_seed(seed)
        with torch.inference_mode():
            out = pipeline(
                image=[person, clothing_image, ref_img],
                prompt=build_prompt("single", use_n_1),
                generator=gen,
                true_cfg_scale=true_cfg_scale,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                width=704,
                height=1248,
            )

        out_raw = out.images[0]
        out_path, out_for_ref = save_and_align(img_path, out_raw)
        pred_by_idx[idx] = out_for_ref
        generated.add(idx)

        post_eval_and_log(img_path, out_path, out_raw, use_n1_value=1)
        clear_gpu_cache()
        return out_for_ref

    def gen_merge(idx: int, ref_left: Image.Image, ref_right: Image.Image, tag: str) -> Image.Image:
        img_path = image_files[idx]
        person = Image.open(img_path).convert("RGB")
        log_inputs(person, img_path, ref_kind=f"merge_{tag}", ref_a=ref_left, ref_b=ref_right)

        gen = torch.Generator(device="cpu").manual_seed(seed)

        try:
            with torch.inference_mode():
                out = pipeline(
                    image=[person, clothing_image, ref_left, ref_right],
                    prompt=build_prompt("both", use_n_1),
                    generator=gen,
                    true_cfg_scale=true_cfg_scale,
                    negative_prompt=negative_prompt,
                    num_inference_steps=num_inference_steps,
                    width=704,
                    height=1248,
                )
        except Exception:
            ref_combined = concat_refs_side_by_side(ref_left, ref_right)
            with torch.inference_mode():
                out = pipeline(
                    image=[person, clothing_image, ref_combined],
                    prompt=build_prompt("both", use_n_1),
                    generator=gen,
                    true_cfg_scale=true_cfg_scale,
                    negative_prompt=negative_prompt,
                    num_inference_steps=num_inference_steps,
                    width=704,
                    height=1248,
                )
            wandb.log({"qwen/ref_combined_fallback": wandb.Image(ref_combined, caption=f"{img_path.stem}_combined_ref")})

        out_raw = out.images[0]
        out_path, out_for_ref = save_and_align(img_path, out_raw)
        pred_by_idx[idx] = out_for_ref
        generated.add(idx)

        post_eval_and_log(img_path, out_path, out_raw, use_n1_value=1)
        clear_gpu_cache()
        return out_for_ref

    front_ref_right = gen_solo(front_idx, tag="front_seed")
    front_ref_left = front_ref_right

    if back_idx not in generated:
        back_ref_left = gen_solo(back_idx, tag="back_seed")
        back_ref_right = back_ref_left
    else:
        back_ref_left = pred_by_idx[back_idx]
        back_ref_right = pred_by_idx[back_idx]

    f_r = front_idx + 1
    f_l = n - 1
    b_l = back_idx - 1
    b_r = back_idx + 1

    def has_both_neighbors(idx: int) -> bool:
        if idx - 1 < 0 or idx + 1 >= n:
            return False
        return (idx - 1) in generated and (idx + 1) in generated

    #schedule: front-right, front-left, back-left, back-right
    order = ["f_r", "f_l", "b_l", "b_r"]

    def next_candidate(which: str) -> int | None:
        nonlocal f_r, f_l, b_l, b_r

        if which == "f_r":
            while f_r < n and f_r in generated:
                f_r += 1
            return f_r if f_r < n else None

        if which == "f_l":
            while f_l >= 0 and f_l in generated:
                f_l -= 1
            return f_l if f_l >= 0 else None

        if which == "b_l":
            while b_l >= 0 and b_l in generated:
                b_l -= 1
            return b_l if b_l >= 0 else None

        if which == "b_r":
            while b_r < n and b_r in generated:
                b_r += 1
            return b_r if b_r < n else None

        return None

    # main loop until all generated
    while len(generated) < n:
        progressed = False

        for which in order:
            idx = next_candidate(which)
            if idx is None or idx in generated:
                continue

            if has_both_neighbors(idx):
                left_ref = pred_by_idx[idx - 1]
                right_ref = pred_by_idx[idx + 1]
                gen_merge(idx, left_ref, right_ref, tag=f"{idx-1}_{idx+1}")
                progressed = True

                # advance pointer used
                if which == "f_r":
                    f_r += 1
                elif which == "f_l":
                    f_l -= 1
                elif which == "b_l":
                    b_l -= 1
                elif which == "b_r":
                    b_r += 1
                continue

            #otherwise single-step from its chain ref
            if which == "f_r":
                front_ref_right = gen_single(idx, front_ref_right, tag="front_right")
                f_r += 1
                progressed = True
                continue

            if which == "f_l":
                front_ref_left = gen_single(idx, front_ref_left, tag="front_left")
                f_l -= 1
                progressed = True
                continue

            if which == "b_l":
                back_ref_left = gen_single(idx, back_ref_left, tag="back_left")
                b_l -= 1
                progressed = True
                continue

            if which == "b_r":
                back_ref_right = gen_single(idx, back_ref_right, tag="back_right")
                b_r += 1
                progressed = True
                continue

        if not progressed:
            #safety fallback: find any remaining idx and generate from nearest available neighbor
            remaining = [i for i in range(n) if i not in generated]
            if not remaining:
                break
            idx = remaining[0]

            if has_both_neighbors(idx):
                gen_merge(idx, pred_by_idx[idx - 1], pred_by_idx[idx + 1], tag="fallback_merge")
            else:
                # choose any available neighbor as ref
                ref = None
                if (idx - 1) in generated:
                    ref = pred_by_idx[idx - 1]
                elif (idx + 1) in generated:
                    ref = pred_by_idx[idx + 1]
                else:
                    # extreme edge case (shouldn't happen): solo
                    ref = None

                if ref is None:
                    gen_solo(idx, tag="fallback_solo")
                else:
                    gen_single(idx, ref, tag="fallback_single")


def main():
    args = parse_args()
    qwen_cfg = {
        "model_path": args.model_path,
        "source_dir": args.source_dir,
        "clothing_image": args.clothing_image,
        "output_dir": args.output_dir,
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "true_cfg_scale": args.true_cfg_scale,
        "num_inference_steps": args.num_inference_steps,
        "seed": args.seed,
        "extensions": args.extensions,
        "use_lora": args.use_lora,
        "lora_scale": args.lora_scale,
        # keep n-1 controllable from yaml
        "use_n_1": True,
    }
    run_qwen_from_config_dict(qwen_cfg)


if __name__ == "__main__":
    main()