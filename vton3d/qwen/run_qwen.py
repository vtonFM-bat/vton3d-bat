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
    """
    Parse command line arguments for the Qwen Image Edit batch processor.
    """
    parser = argparse.ArgumentParser(description="Batch Qwen Image Edit pipeline.")

    parser.add_argument(
        "--model_path",
        type=str,
        default="Qwen/Qwen-Image-Edit-2511",
        help="Path or HuggingFace model ID."
    )
    parser.add_argument(
        "--source_dir",
        type=str,
        required=True,
        help="Directory containing person images."
    )
    parser.add_argument(
        "--clothing_image",
        type=str,
        required=True,
        help="Path to the clothing image that should replace the original clothing."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the edited images."
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=(
            "Replace the person's original clothing with the clothing image. "
            "Do not invent new features. Keep the person identical: pose, body, "
            "face, hair, and background must remain unchanged."
        ),
        help="Editing prompt."
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="change pose",
        help="Negative prompt."
    )
    parser.add_argument(
        "--true_cfg_scale",
        type=float,
        default=4.0,
        help="CFG scale value."
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=20,
        help="Number of inference steps."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed."
    )
    parser.add_argument(
        "--extensions",
        type=str,
        nargs="+",
        default=[".jpg", ".jpeg", ".png"],
        help="File extensions treated as images."
    )

    parser.add_argument(
        "--use_lora",
        type=str,
        default="",
        help="Path to LoRA .safetensors (empty => disable)."
    )
    parser.add_argument(
        "--lora_scale",
        type=float,
        default=1.0,
        help="LoRA scale (only if use_lora is set)."
    )

    return parser.parse_args()


def infer_eval_flag_from_clothing_path(clothing_path: Path) -> str:
    """
    Infer eval_flag ("upper" or "lower") from clothing image path.
    Expects the path to contain a folder named 'upper' or 'lower'.
    """
    parts = [p.lower() for p in clothing_path.parts]

    if "dress" in parts:
        return "dress"

    if "lower" in parts and "upper" in parts:
        raise ValueError(
            f"Ambiguous clothing path contains both 'upper' and 'lower': {clothing_path}"
        )

    if "lower" in parts:
        return "lower"
    if "upper" in parts:
        return "upper"

    raise ValueError(
        f"Could not infer eval_flag from clothing path (missing 'upper'/'lower' folder): {clothing_path}"
    )


def infer_length_flag_from_clothing_path(clothing_path: Path) -> str:
    parts = [p.lower() for p in clothing_path.parts]

    if "long" in parts and "short" in parts:
        raise ValueError(f"Ambiguous clothing path contains both 'long' and 'short': {clothing_path}")

    if "long" in parts:
        return "long"
    if "short" in parts:
        return "short"

    raise ValueError(
        f"Could not infer length_flag from clothing path (missing 'long'/'short' folder): {clothing_path}"
    )


from pathlib import Path

def load_pipeline(model_path: str, use_lora: str = "", lora_scale: float = 1.0) -> QwenImageEditPlusPipeline:
    """
    Load the Qwen Image Edit pipeline with CPU offload enabled.
    Optionally load a LoRA adapter from a .safetensors file or a directory/repo.
    """
    from safetensors.torch import load_file, save_file
    import tempfile
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
    )
    pipe.set_progress_bar_config(disable=None)
    pipe.enable_model_cpu_offload()

    # LoRA optional
    use_lora = (use_lora or "").strip()
    if use_lora != "":
        lora_path = Path(use_lora).expanduser().resolve()
        adapter_name = "lora"

        if not lora_path.exists():
            raise FileNotFoundError(f"LoRA path not found: {lora_path}")

        if lora_path.is_file() and lora_path.suffix == ".safetensors":
            sd = load_file(str(lora_path))

            #remove extra training heads
            drop_prefixes = ("vggt_loss.",)

            keys_before = list(sd.keys())
            sd_filtered = {k: v for k, v in sd.items() if not k.startswith(drop_prefixes)}

            dropped = [k for k in keys_before if k not in sd_filtered]
            kept = len(sd_filtered)

            if kept == 0:
                raise ValueError(
                    f"After filtering, no keys left in LoRA file. "
                    f"Dropped={len(dropped)}. Example dropped: {dropped[:5]}"
                )

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
            pipe.load_lora_weights(
                str(lora_path),
                adapter_name=adapter_name,
            )

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
    """
    Extract trailing number from filename like 'florian_0150.jpg' -> 150
    If not found, returns a large number so it sorts last.
    """
    m = re.search(r"_([0-9]+)\.[^.]+$", p.name)
    if not m:
        return 10**18
    return int(m.group(1))


def get_image_files(source_dir: Path, extensions: list[str]) -> list[Path]:
    """
    Return all image files in a directory matching the allowed extensions,
    sorted by numeric frame index extracted from filename.
    """
    exts = {e.lower() for e in extensions}
    files = [
        p for p in source_dir.iterdir()
        if p.is_file() and p.suffix.lower() in exts
    ]
    return sorted(files, key=_extract_frame_number)


def concat_refs_side_by_side(img_left: Image.Image, img_right: Image.Image) -> Image.Image:
    """
    Fallback if pipeline doesn't accept 4 images:
    create a single reference image by concatenating left/right horizontally.
    """
    img_left = img_left.convert("RGB")
    img_right = img_right.convert("RGB")

    h = max(img_left.height, img_right.height)
    w = img_left.width + img_right.width

    canvas = Image.new("RGB", (w, h))
    canvas.paste(img_left, (0, 0))
    canvas.paste(img_right, (img_left.width, 0))
    return canvas


def clear_gpu_cache():
    """
    Clear GPU and CPU memory caches after each image.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def align_and_overwrite_with_optical_flow(
    aligner,
    src_path: Path,
    tgt_path: Path,
    composite_original_outside_mask: bool,
):
    """
    Runs MaskedOpticalFlow alignment (src->tgt), overwrites src_path in-place,
    and returns debug info for logging.
    """
    result = aligner.run_from_paths(
        src_path=src_path,
        tgt_path=tgt_path,
        output_path=src_path,  # overwrite
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

    diff2 = diff ** 2
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
    """
    Runs optical flow to align out_path -> real_path, overwrites out_path,
    logs to wandb under opticalflow/*, returns aligned PIL.Image.
    """
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
    """
    Run the Qwen clothing edit batch using a config dictionary (e.g. cfg['qwen']).

    Supports optional per-clothing-type prompts via a `prompts` dict in the config.
    """
    wandb.define_metric("qwen/*", step_metric="qwen/image_index")

    model_path = qwen_cfg.get("model_path", "Qwen/Qwen-Image-Edit-2511")
    source_dir = Path(qwen_cfg["source_dir"])
    clothing_path = Path(qwen_cfg["clothing_image"])
    output_dir = Path(qwen_cfg["output_dir"])

    default_prompt = qwen_cfg.get(
        "prompt",
        (
            "Replace the person's original clothing with the clothing image. "
            "Do not invent new features. Keep the person identical: pose, body, "
            "face, hair, and background must remain unchanged."
        ),
    )
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

    if not image_files:
        raise RuntimeError(f"No images found in {source_dir}.")

    frame_nums = [_extract_frame_number(p) for p in image_files]
    path_by_frame = {fn: p for fn, p in zip(frame_nums, image_files)}
    sorted_frames = sorted(path_by_frame.keys())
    print(f"[frames] count={len(sorted_frames)} first={sorted_frames[:5]} last={sorted_frames[-5:]}")

    front_frame = sorted_frames[0]
    last_frame = sorted_frames[-1]
    front_img_path = path_by_frame[front_frame]

    first_num = _extract_frame_number(image_files[0])
    if first_num != 0:
        print(
            f"[WARN] expected front frame number 0, "
            f"got {first_num} for {image_files[0].name}"
        )

    if len(image_files) > 1:
        print("[order check]", image_files[0].name, image_files[1].name, image_files[-1].name)
    else:
        print("[order check]", image_files[0].name)

    use_lora = qwen_cfg.get("use_lora", "")
    lora_scale = float(qwen_cfg.get("lora_scale", 1.0))
    pipeline = load_pipeline(model_path, use_lora=use_lora, lora_scale=lora_scale)

    img_count = 0
    wandb.log({"qwen/clothing_image": wandb.Image(clothing_image, caption=clothing_path.name)})

    estimator = SapiensSegmentation(
        SapiensSegmentationType.SEGMENTATION_1B,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        dtype=torch.float16,
    )

    eval_flag = infer_eval_flag_from_clothing_path(clothing_path)

    if eval_flag == "dress":
        length_flag = None
    else:
        length_flag = infer_length_flag_from_clothing_path(clothing_path)

    print(f"[qwen] inferred eval_flag='{eval_flag}', length_flag='{length_flag}' from clothing_image='{clothing_path}'")

    prompts_map = qwen_cfg.get("prompts", {}) or {}
    negative_prompts_map = qwen_cfg.get("negative_prompts", {}) or {}

    base_prompt = prompts_map.get(eval_flag, default_prompt)
    print(f"Prompt:{base_prompt}")

    def tweak_base_prompt_for_n1_non_front(p: str) -> str:
        """
        For use_n_1 == True and ref_kind != 'front':
        change "... all other clothes of the person, lighting ..."  ->
               "... all other clothes of the person in image 1, lighting ..."
        """
        needle = "Keep background, pose, face, hair, skin, body shape, all other clothes of the person,"
        repl   = "Keep background, pose, face, hair, skin, body shape, all other clothes of the person in image 1,"
        if needle in p:
            return p.replace(needle, repl, 1)
        return p

    negative_prompt = negative_prompts_map.get(eval_flag, default_negative_prompt)
    print(f"negative Prompt: {negative_prompt}")

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
        """
        ref_kind:
          - 'front'  -> image=[person, clothing]
          - 'single' -> image=[person, clothing, ref]
          - 'both'   -> image=[person, clothing, ref_left, ref_right]
        """
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

    use_n_1 = bool(qwen_cfg.get("use_n_1", False))
    solo_gen_every = int(qwen_cfg.get("solo_gen_every", 0) or 0)
    n = len(image_files)

    predicted_cache: dict[int, Image.Image] = {}
    img_count = 0

    def log_neighbor_fashionclip(curr_img_path: Path, curr_out_path: Path):
        """
        Online neighbor eval:
          - compare output(f) vs output(f-1) if output(f-1) already exists
          - front has no previous -> log NaN
          - last frame: additionally compare output(last) vs output(front)
        """
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
                    f"qwen/masked_neighbor_prev_{eval_flag}": wandb.Image(
                        m_prev, caption=f"{prev_out_path.stem}_masked_prev"
                    ),
                    f"qwen/masked_neighbor_curr_{eval_flag}": wandb.Image(
                        m_curr, caption=f"{curr_out_path.stem}_masked_curr"
                    ),
                })
            else:
                wandb.log({f"qwen/fc_neighbor_prev_{eval_flag}_{length_flag}": float("nan")})

        if f == last_frame:
            front_out_path = output_dir / f"{front_img_path.stem}.png"
            if front_out_path.exists():
                sim_ff, m_front, m_last = qwen_fashionclip_similarity_neighbor_masked_clothing(
                    person_img_a_path=str(front_out_path),
                    person_img_b_path=str(curr_out_path),
                    flag=eval_flag,
                    estimator=estimator,
                    clip_device="cuda",
                    return_masked_rgb=True,
                )
                wandb.log({
                    f"qwen/fc_neighbor_front_{eval_flag}_{length_flag}": sim_ff,
                    f"qwen/masked_neighbor_front_{eval_flag}": wandb.Image(
                        m_front, caption=f"{front_out_path.stem}_masked_front"
                    ),
                    f"qwen/masked_neighbor_last_{eval_flag}": wandb.Image(
                        m_last, caption=f"{curr_out_path.stem}_masked_last"
                    ),
                })
            else:
                wandb.log({f"qwen/fc_neighbor_front_{eval_flag}_{length_flag}": float("nan")})

    if not use_n_1:
        for idx, img_path in enumerate(image_files):
            person_image = Image.open(img_path).convert("RGB")
            generator = torch.Generator(device="cpu").manual_seed(seed)

            inputs = {
                "image": [person_image, clothing_image],
                "prompt": build_prompt("front", use_n_1),
                "generator": generator,
                "true_cfg_scale": true_cfg_scale,
                "negative_prompt": negative_prompt,
                "num_inference_steps": num_inference_steps,
                "width": 704,
                "height": 1248,
            }
            with torch.inference_mode():
                output = pipeline(**inputs)

            output_image = output.images[0]
            out_path = output_dir / f"{img_path.stem}.png"
            output_image.save(out_path)

            # neighbor eval (sequential by filename number)
            log_neighbor_fashionclip(curr_img_path=img_path, curr_out_path=out_path)

            mse_value, psnr_value, heatmap = qwen_eval_masked(
                img1_path=str(img_path),
                img2_path=str(out_path),
                flag=eval_flag,
                length_flag=length_flag,
                estimator=estimator,
            )

            face_sim, face_in_rgb, face_out_rgb = qwen_arcface_similarity_input_vs_output(
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
                "qwen/input_image": wandb.Image(person_image, caption=img_path.name),
                "qwen/output_image": wandb.Image(output_image, caption=out_path.name),
                "qwen/image_index": img_count,
                "qwen/use_n_1": 0,
                f"qwen/mse_non_clothed_area_{eval_flag}_{length_flag}": mse_value,
                f"qwen/psnr_non_clothed_area_{eval_flag}_{length_flag}": psnr_value,
                f"qwen/heatmap_non_clothed_area_{eval_flag}_{length_flag}": wandb.Image(
                    heatmap, caption=f"{img_path.stem}_heatmap_{eval_flag}_{length_flag}"
                ),
                f"qwen/fashionclip_sim_input_{eval_flag}_clothing": fc_sim,
                f"qwen/masked_input_clothing_{eval_flag}": wandb.Image(
                    masked_rgb, caption=f"{img_path.stem}_masked_input_{eval_flag}"
                ),
                "qwen/face_sim_input_vs_output": face_sim,
            })

            clear_gpu_cache()

    else:
        if n == 0:
            return

        front_idx = 0
        front_path = image_files[front_idx]
        front_person = Image.open(front_path).convert("RGB")
        generator = torch.Generator(device="cpu").manual_seed(seed)

        with torch.inference_mode():
            output = pipeline(
                image=[front_person, clothing_image],
                prompt=build_prompt("front", use_n_1),
                generator=generator,
                true_cfg_scale=true_cfg_scale,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                width=704,
                height=1248,
            )

        front_out_raw = output.images[0]
        front_out_path = output_dir / f"{front_path.stem}.png"
        front_out_raw.save(front_out_path)

        img_count += 1
        wandb.log({
            "qwen/input_image": wandb.Image(front_person, caption=front_path.name),
            "qwen/output_image": wandb.Image(front_out_raw, caption=front_out_path.name),
            "qwen/image_index": img_count,
            "qwen/use_n_1": 1,
            "qwen/ref_kind": "none_front",
        })

        front_for_ref = front_out_raw

        if aligner is not None:
            try:
                front_for_ref = run_mof_and_get_aligned_pil(
                    aligner=aligner,
                    out_path=front_out_path,
                    real_path=Path(front_path),
                    composite_original_outside_mask=composite_original_outside_mask,
                )
            except Exception as e:
                print(f"[WARN] optical flow failed for {front_out_path.name}: {e}")

        # neighbor eval for front (prev=NaN)
        log_neighbor_fashionclip(curr_img_path=front_path, curr_out_path=front_out_path)

        predicted_cache[front_idx] = front_for_ref
        clear_gpu_cache()

        l, r = 1, n - 1
        ref_right = front_for_ref
        ref_left = front_for_ref

        toggle = True
        while l < r:
            if toggle:
                idx = l
                ref = ref_right
                l += 1
                side = "right"
            else:
                idx = r
                ref = ref_left
                r -= 1
                side = "left"
            toggle = not toggle

            img_path = image_files[idx]

            frame_number = _extract_frame_number(img_path)

            pos = bisect.bisect_left(sorted_frames, frame_number)
            do_solo = (
                    solo_gen_every > 0
                    and frame_number != front_frame
                    and pos % solo_gen_every == 0
            )

            person_image = Image.open(img_path).convert("RGB")
            generator = torch.Generator(device="cpu").manual_seed(seed)

            wandb.log({
                "qwen/input_image": wandb.Image(person_image, caption=img_path.name),
                "qwen/clothing_image": wandb.Image(clothing_image, caption=clothing_path.name),
                "qwen/use_n_1": 1,
                f"qwen/ref_{side}": wandb.Image(ref, caption=f"ref_for_{img_path.stem}"),
                "qwen/ref_kind": f"single_{side}",
            })

            if do_solo:
                print(f"[qwen] SOLO generation for frame {frame_number}")
                with torch.inference_mode():
                    output = pipeline(
                        image=[person_image, clothing_image],
                        prompt=build_prompt("front", use_n_1),
                        generator=generator,
                        true_cfg_scale=true_cfg_scale,
                        negative_prompt=negative_prompt,
                        num_inference_steps=num_inference_steps,
                        width=704,
                        height=1248,
                    )
            else:
                with torch.inference_mode():
                    output = pipeline(
                        image=[person_image, clothing_image, ref],
                        prompt=build_prompt("single", use_n_1),
                        generator=generator,
                        true_cfg_scale=true_cfg_scale,
                        negative_prompt=negative_prompt,
                        num_inference_steps=num_inference_steps,
                        width=704,
                        height=1248,
                    )

            out_img_raw = output.images[0]
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

            # neighbor eval (sequential by filename number)
            log_neighbor_fashionclip(curr_img_path=img_path, curr_out_path=out_path)

            predicted_cache[idx] = out_img_for_ref

            if side == "right":
                ref_right = out_img_for_ref
            else:
                ref_left = out_img_for_ref

            mse_value, psnr_value, heatmap = qwen_eval_masked(
                img1_path=str(img_path),
                img2_path=str(out_path),
                flag=eval_flag,
                length_flag=length_flag,
                estimator=estimator,
            )

            face_sim, face_in_rgb, face_out_rgb = qwen_arcface_similarity_input_vs_output(
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
                "qwen/use_n_1": 1,
                f"qwen/mse_non_clothed_area_{eval_flag}_{length_flag}": mse_value,
                f"qwen/psnr_non_clothed_area_{eval_flag}_{length_flag}": psnr_value,
                f"qwen/heatmap_non_clothed_area_{eval_flag}_{length_flag}": wandb.Image(
                    heatmap, caption=f"{img_path.stem}_heatmap_{eval_flag}_{length_flag}"
                ),
                f"qwen/fashionclip_sim_input_{eval_flag}_clothing": fc_sim,
                f"qwen/masked_input_clothing_{eval_flag}": wandb.Image(
                    masked_rgb, caption=f"{img_path.stem}_masked_input_{eval_flag}"
                ),
                "qwen/face_sim_input_vs_output": face_sim,
            })

            clear_gpu_cache()

        if l == r and n > 1:
            idx = l
            img_path = image_files[idx]
            person_image = Image.open(img_path).convert("RGB")
            generator = torch.Generator(device="cpu").manual_seed(seed)

            ref_combined = None
            try:
                with torch.inference_mode():
                    output = pipeline(
                        image=[person_image, clothing_image, ref_left, ref_right],
                        prompt=build_prompt("both", use_n_1),
                        generator=generator,
                        true_cfg_scale=true_cfg_scale,
                        negative_prompt=negative_prompt,
                        num_inference_steps=num_inference_steps,
                        width=704,
                        height=1248,
                    )
            except Exception:
                ref_combined = concat_refs_side_by_side(ref_left, ref_right)
                with torch.inference_mode():
                    output = pipeline(
                        image=[person_image, clothing_image, ref_combined],
                        prompt=build_prompt("both", use_n_1),
                        generator=generator,
                        true_cfg_scale=true_cfg_scale,
                        negative_prompt=negative_prompt,
                        num_inference_steps=num_inference_steps,
                        width=704,
                        height=1248,
                    )

            out_img_raw = output.images[0]
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

            # neighbor eval (sequential by filename number)
            log_neighbor_fashionclip(curr_img_path=img_path, curr_out_path=out_path)

            predicted_cache[idx] = out_img_for_ref

            wandb.log({
                "qwen/input_image": wandb.Image(person_image, caption=img_path.name),
                "qwen/clothing_image": wandb.Image(clothing_image, caption=clothing_path.name),
                "qwen/use_n_1": 1,
                "qwen/ref_left": wandb.Image(ref_left, caption="left_chain_ref"),
                "qwen/ref_right": wandb.Image(ref_right, caption="right_chain_ref"),
                **({"qwen/ref_combined_fallback": wandb.Image(ref_combined, caption="combined_ref_fallback")}
                   if ref_combined is not None else {}),
                "qwen/ref_kind": "both_neighbors",
            })

            mse_value, psnr_value, heatmap = qwen_eval_masked(
                img1_path=str(img_path),
                img2_path=str(out_path),
                flag=eval_flag,
                length_flag=length_flag,
                estimator=estimator,
            )

            face_sim, face_in_rgb, face_out_rgb = qwen_arcface_similarity_input_vs_output(
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
                "qwen/use_n_1": 1,
                f"qwen/mse_non_clothed_area_{eval_flag}_{length_flag}": mse_value,
                f"qwen/psnr_non_clothed_area_{eval_flag}_{length_flag}": psnr_value,
                f"qwen/heatmap_non_clothed_area_{eval_flag}_{length_flag}": wandb.Image(
                    heatmap, caption=f"{img_path.stem}_heatmap_{eval_flag}_{length_flag}"
                ),
                f"qwen/fashionclip_sim_input_{eval_flag}_clothing": fc_sim,
                f"qwen/masked_input_clothing_{eval_flag}": wandb.Image(
                    masked_rgb, caption=f"{img_path.stem}_masked_input_{eval_flag}"
                ),
                "qwen/face_sim_input_vs_output": face_sim,
            })

            clear_gpu_cache()


def main():
    """
    CLI entry point for running Qwen clothing edit from the command line.
    """
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
    }
    run_qwen_from_config_dict(qwen_cfg)


if __name__ == "__main__":
    main()