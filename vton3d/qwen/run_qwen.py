import argparse
from pathlib import Path
import gc
import sys
from PIL import Image
import torch
from diffusers import QwenImageEditPlusPipeline
import wandb

from vton3d.utils.qwen_eval import (
    qwen_eval_masked,
    qwen_fashionclip_similarity_masked_clothing,
    qwen_arcface_similarity_input_vs_output,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SAPIENS_REPO = REPO_ROOT / "Sapiens-Pytorch-Inference"
sys.path.insert(0, str(SAPIENS_REPO))

from sapiens_inference.segmentation import SapiensSegmentation, SapiensSegmentationType
import numpy as np


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
        # sehr selten, aber dann ist der Pfad ambig
        raise ValueError(
            f"Ambiguous clothing path contains both 'upper' and 'lower': {clothing_path}"
        )

    if "lower" in parts:
        return "lower"
    if "upper" in parts:
        return "upper"

    # Optional: falls du lieber defaulten willst statt Fehler:
    # return "upper"
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


def load_pipeline(model_path: str) -> QwenImageEditPlusPipeline:
    """
    Load the Qwen Image Edit pipeline with CPU offload enabled.
    """
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
    )
    pipe.set_progress_bar_config(disable=None)
    pipe.enable_model_cpu_offload()
    return pipe


import re
from pathlib import Path

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


def run_qwen_from_config_dict(qwen_cfg: dict):
    """
    Run the Qwen clothing edit batch using a config dictionary (e.g. cfg['qwen']).

    Supports optional per-clothing-type prompts via a `prompts` dict in the config.

    Example YAML structure:
    qwen:
      prompt: "default prompt"
      prompts:
        upper: "prompt for shirts/upper"
        lower: "prompt for pants/lower"
        dress: "prompt for dresses"
      negative_prompt: "default neg"
      negative_prompts:
        upper: "neg for upper"
        lower: "neg for lower"
        dress: "neg for dress"
    """
    wandb.define_metric("qwen/*", step_metric="qwen/image_index")

    model_path = qwen_cfg.get("model_path", "Qwen/Qwen-Image-Edit-2511")
    source_dir = Path(qwen_cfg["source_dir"])
    clothing_path = Path(qwen_cfg["clothing_image"])
    output_dir = Path(qwen_cfg["output_dir"])

    # basic defaults (these may be overridden per clothing type below)
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

    first_num = _extract_frame_number(image_files[0])
    if first_num != 0:
        print(
            f"[WARN] expected front frame number 0, "
            f"got {first_num} for {image_files[0].name}"
        )

    print("[order check]", image_files[0].name, image_files[1].name, image_files[-1].name)

    pipeline = load_pipeline(model_path)
    base_generator = torch.Generator(device="cpu").manual_seed(seed)
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

    # choose per-clothing prompts if provided, otherwise fall back to defaults
    prompts_map = qwen_cfg.get("prompts", {}) or {}
    negative_prompts_map = qwen_cfg.get("negative_prompts", {}) or {}

    prompt = prompts_map.get(eval_flag, default_prompt)
    print(f"Prompt:{prompt}")
    negative_prompt = negative_prompts_map.get(eval_flag, default_negative_prompt)
    print(f"negative Prompt: {negative_prompt}")

    use_n_1 = bool(qwen_cfg.get("use_n_1", False))
    n = len(image_files)

    predicted_cache: dict[int, Image.Image] = {}

    img_count = 0

    if not use_n_1:
        for idx, img_path in enumerate(image_files):
            person_image = Image.open(img_path).convert("RGB")
            generator = torch.Generator(device="cpu").manual_seed(seed)

            inputs = {
                "image": [person_image, clothing_image],
                "prompt": prompt,
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
                prompt=prompt,
                generator=generator,
                true_cfg_scale=true_cfg_scale,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                width=704,
                height=1248,
            )

        front_out = output.images[0]
        front_out_path = output_dir / f"{front_path.stem}.png"
        front_out.save(front_out_path)
        predicted_cache[front_idx] = front_out

        img_count += 1
        wandb.log({
            "qwen/input_image": wandb.Image(front_person, caption=front_path.name),
            "qwen/output_image": wandb.Image(front_out, caption=front_out_path.name),
            "qwen/image_index": img_count,
            "qwen/use_n_1": 1,
            "qwen/ref_kind": "none_front",
        })

        clear_gpu_cache()

        l, r = 1, n - 1
        ref_right = front_out
        ref_left = front_out

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
            person_image = Image.open(img_path).convert("RGB")
            generator = torch.Generator(device="cpu").manual_seed(seed)

            wandb.log({
                "qwen/input_image": wandb.Image(person_image, caption=img_path.name),
                "qwen/clothing_image": wandb.Image(clothing_image, caption=clothing_path.name),
                "qwen/use_n_1": 1,
                f"qwen/ref_{side}": wandb.Image(ref, caption=f"ref_for_{img_path.stem}"),
                "qwen/ref_kind": f"single_{side}",
            })

            with torch.inference_mode():
                output = pipeline(
                    image=[person_image, clothing_image, ref],
                    prompt=prompt,
                    generator=generator,
                    true_cfg_scale=true_cfg_scale,
                    negative_prompt=negative_prompt,
                    num_inference_steps=num_inference_steps,
                    width=704,
                    height=1248,
                )

            out_img = output.images[0]
            out_path = output_dir / f"{img_path.stem}.png"
            out_img.save(out_path)
            predicted_cache[idx] = out_img

            if side == "right":
                ref_right = out_img
            else:
                ref_left = out_img

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
                "qwen/output_image": wandb.Image(out_img, caption=out_path.name),
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
                        prompt=prompt,
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
                        prompt=prompt,
                        generator=generator,
                        true_cfg_scale=true_cfg_scale,
                        negative_prompt=negative_prompt,
                        num_inference_steps=num_inference_steps,
                        width=704,
                        height=1248,
                    )

            out_img = output.images[0]
            out_path = output_dir / f"{img_path.stem}.png"
            out_img.save(out_path)
            predicted_cache[idx] = out_img

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
                "qwen/output_image": wandb.Image(out_img, caption=out_path.name),
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
    }
    run_qwen_from_config_dict(qwen_cfg)


if __name__ == "__main__":
    main()
