import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import wandb
from diffusers import QwenImageEditPlusPipeline


def parse_args():
    p = argparse.ArgumentParser("Qwen seed stability test (wandb + mse)")

    # required
    p.add_argument("--person_image", required=True, type=str)
    p.add_argument("--clothing_image", required=True, type=str)

    # model / io
    p.add_argument("--model_path", default="Qwen/Qwen-Image-Edit-2511", type=str)
    p.add_argument("--out_dir", default="seed_test_out", type=str)

    # prompts
    p.add_argument("--prompt", default=(
        "Replace the person's original clothing with the clothing image. "
        "Do not invent new features. Keep the person identical: pose, body, "
        "face, hair, and background must remain unchanged."
    ), type=str)
    p.add_argument("--negative_prompt", default="change pose", type=str)

    # diffusion params
    p.add_argument("--true_cfg_scale", type=float, default=4.0)
    p.add_argument("--num_inference_steps", type=int, default=20)
    p.add_argument("--width", type=int, default=704)
    p.add_argument("--height", type=int, default=1248)

    # seed sweep
    p.add_argument("--num_samples", type=int, default=10)
    p.add_argument("--seed_start", type=int, default=0)

    # wandb
    p.add_argument("--wandb_project", default="vton_pipeline", type=str)
    p.add_argument("--wandb_run_name", default=None, type=str)
    p.add_argument("--wandb_mode", default=None, type=str, help="online|offline|disabled")

    # logging options
    p.add_argument("--log_images", action="store_true", help="log input/output images to wandb")

    return p.parse_args()


def load_pipeline(model_path: str) -> QwenImageEditPlusPipeline:
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
    )
    pipe.enable_model_cpu_offload()
    return pipe


def pil_to_float01_rgb(img: Image.Image) -> np.ndarray:
    """PIL RGB -> float32 [H,W,3] in [0,1]."""
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    return arr


def mse_rgb(a01: np.ndarray, b01: np.ndarray) -> float:
    """Mean squared error over all pixels/channels, both float [0,1]."""
    diff = a01 - b01
    return float(np.mean(diff * diff))


def main():
    args = parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # wandb init
    if args.wandb_mode is not None:
        # wandb reads this env var too, but init(mode=...) is enough in most setups
        wandb.init(project=args.wandb_project, name=args.wandb_run_name, mode=args.wandb_mode, config=vars(args))
    else:
        wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))

    person_img_orig = Image.open(args.person_image).convert("RGB")
    clothing_img = Image.open(args.clothing_image).convert("RGB")

    # log reference clothing once
    if args.log_images:
        wandb.log({
            "qwen/person_image_ref": wandb.Image(person_img_orig, caption=Path(args.person_image).name),
            "qwen/clothing_image_ref": wandb.Image(clothing_img, caption=Path(args.clothing_image).name),
        })

    pipe = load_pipeline(args.model_path)

    for idx in range(1, args.num_samples + 1):
        seed = args.seed_start + (idx - 1)
        gen = torch.Generator(device="cpu").manual_seed(seed)

        out = pipe(
            image=[person_img_orig, clothing_img],
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            true_cfg_scale=args.true_cfg_scale,
            num_inference_steps=args.num_inference_steps,
            generator=gen,
            width=args.width,
            height=args.height,
        )

        out_img = out.images[0]

        # save
        save_path = out_dir / f"seed_{seed:05d}.png"
        out_img.save(save_path)

        # MSE: resize input person to output size for fair pixelwise compare
        person_resized = person_img_orig.resize(out_img.size, Image.BICUBIC)
        mse = mse_rgb(pil_to_float01_rgb(out_img), pil_to_float01_rgb(person_resized))

        log_dict = {
            "qwen/image_index": idx,
            "qwen/seed": seed,
            "qwen/mse_output_vs_input": mse,
        }
        if args.log_images:
            log_dict.update({
                "qwen/output_image": wandb.Image(out_img, caption=f"seed={seed}"),
            })

        wandb.log(log_dict, step=idx)
        print(f"[{idx}/{args.num_samples}] seed={seed} saved={save_path} mse={mse:.6f}")

    wandb.finish()
    print("done.")


if __name__ == "__main__":
    main()
