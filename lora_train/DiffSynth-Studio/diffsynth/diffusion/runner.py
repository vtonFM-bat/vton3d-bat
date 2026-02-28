import os, torch
from tqdm import tqdm
from accelerate import Accelerator
from .training_module import DiffusionTrainingModule
from .logger import ModelLogger
from pathlib import Path


def launch_training_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    learning_rate: float = 1e-5,
    weight_decay: float = 1e-2,
    num_workers: int = 1,
    save_steps: int = None,
    num_epochs: int = 1,
    args=None,
    val_dataset=None,
):
    if args is not None:
        learning_rate = args.learning_rate
        weight_decay = args.weight_decay
        num_workers = args.dataset_num_workers
        save_steps = args.save_steps
        num_epochs = args.num_epochs

    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, collate_fn=lambda x: x[0], num_workers=num_workers)

    model.to(device=accelerator.device)
    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)

    import json
    from pathlib import Path

    val_dataloader = None
    tracked_val_samples = None

    if val_dataset is not None:
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset, shuffle=False, collate_fn=lambda x: x[0], num_workers=num_workers
        )
        val_dataloader = accelerator.prepare(val_dataloader)

        if args is not None and getattr(args, "val_dataset_metadata_path", None):
            with open(args.val_dataset_metadata_path, "r", encoding="utf-8") as f:
                val_meta = json.load(f)

            if getattr(args, "eval_sample_ids", ""):
                wanted = [s.strip() for s in args.eval_sample_ids.split(",") if s.strip()]
                tracked_val_samples = [s for s in val_meta if s.get("sample_id") in wanted]
            else:
                n = int(getattr(args, "eval_num_samples", 5))
                tracked_val_samples = val_meta[:n]

    def run_validation_loss():
        if val_dataloader is None:
            return None

        model.eval()
        total = 0.0
        n = 0
        max_batches = int(getattr(args, "eval_max_val_batches", 0) or 0) if args is not None else 0

        with torch.no_grad():
            for bi, vdata in enumerate(val_dataloader):
                if max_batches and bi >= max_batches:
                    break
                if val_dataset.load_from_cache:
                    out = model({}, inputs=vdata)
                    vloss = out[0] if isinstance(out, (tuple, list)) else out
                else:
                    out = model(vdata)
                    vloss = out[0] if isinstance(out, (tuple, list)) else out
                total += vloss.detach().float().item()
                n += 1

        model.train()
        return total / max(n, 1)

    def run_tracked_predictions(step: int):
        if (tracked_val_samples is None) or (len(tracked_val_samples) == 0):
            return
        if not accelerator.is_main_process:
            return
        if not os.environ.get("WANDB_PROJECT"):
            return

        from PIL import Image
        import wandb
        import numpy as np
        import matplotlib.cm as cm

        infer_steps = int(getattr(args, "eval_infer_steps", 20)) if args is not None else 20
        cfg_scale = float(getattr(args, "eval_cfg_scale", 1.0)) if args is not None else 1.0
        seed = int(getattr(args, "eval_seed", 0)) if args is not None else 0
        base = Path(getattr(args, "val_dataset_base_path")) if args is not None else Path(".")

        def pil_to_np01(img: Image.Image) -> np.ndarray:
            return np.asarray(img.convert("RGB")).astype(np.float32) / 255.0

        def compute_psnr(tgt01: np.ndarray, pred01: np.ndarray, mask_exclude: np.ndarray | None = None) -> float:
            diff = tgt01 - pred01
            if mask_exclude is not None:
                diff = diff.copy()
                diff[mask_exclude] = 0.0
                denom = (~mask_exclude).sum()
                if denom <= 0:
                    return float("nan")
                mse = (diff ** 2).sum() / (denom * 3.0)
            else:
                mse = float(np.mean(diff ** 2))

            if mse <= 1e-12:
                return 99.0
            return float(10.0 * np.log10(1.0 / mse))

        def make_heatmap_pil(tgt01: np.ndarray, pred01: np.ndarray,
                             mask_exclude: np.ndarray | None = None) -> Image.Image:
            abs_diff = np.abs(tgt01 - pred01)
            heatmap_gray = abs_diff.mean(axis=2)  # HxW
            if mask_exclude is not None:
                heatmap_gray = heatmap_gray.copy()
                heatmap_gray[mask_exclude] = 0.0

            norm = heatmap_gray / (heatmap_gray.max() + 1e-8)
            colormap = cm.get_cmap("Reds")
            heat_rgb = (colormap(norm)[..., :3] * 255).astype(np.uint8)
            return Image.fromarray(heat_rgb, mode="RGB")

        model.eval()
        try:
            with torch.no_grad():
                log_payload = {}
                psnr_vals = []

                for s in tracked_val_samples:
                    sid = s.get("sample_id", "sample")

                    target_path = base / s["image"]
                    edit_paths = [base / p for p in s["edit_image"]]
                    if len(edit_paths) == 0:
                        continue

                    edit_imgs = [Image.open(p).convert("RGB") for p in edit_paths]
                    target_img = Image.open(target_path).convert("RGB")

                    w, h = target_img.size
                    target_img_r = target_img

                    print("before infer timesteps:", len(getattr(model.pipe.scheduler, "timesteps", [])))

                    try:
                        out = model.pipe(
                            s["prompt"],
                            edit_image=edit_imgs,
                            height=h,
                            width=w,
                            num_inference_steps=infer_steps,
                            seed=seed,
                            edit_image_auto_resize=True,
                            zero_cond_t=True,
                            cfg_scale=cfg_scale,
                        )
                    except TypeError:
                        out = model.pipe(
                            s["prompt"],
                            edit_image=edit_imgs,
                            height=h,
                            width=w,
                            num_inference_steps=infer_steps,
                            seed=seed,
                            edit_image_auto_resize=True,
                            zero_cond_t=True,
                        )

                    if hasattr(out, "save"):
                        pred_img = out
                    elif isinstance(out, dict) and "images" in out:
                        pred_img = out["images"][0]
                    else:
                        pred_img = out[0]

                    pred_img_r = pred_img.convert("RGB").resize((w, h))

                    tgt01 = pil_to_np01(target_img_r)
                    pred01 = pil_to_np01(pred_img_r)

                    psnr_val = compute_psnr(tgt01, pred01, mask_exclude=None)
                    psnr_vals.append(psnr_val)

                    heat = make_heatmap_pil(tgt01, pred01, mask_exclude=None).resize((w, h))

                    log_payload[f"val/input/{sid}"] = [
                        wandb.Image(img.convert("RGB").resize((w, h)), caption=f"{sid} edit[{ei}] @ step {step}")
                        for ei, img in enumerate(edit_imgs)
                    ]

                    log_payload[f"val/pred/{sid}"] = wandb.Image(
                        pred_img_r, caption=f"{sid} pred @ step {step} | PSNR={psnr_val:.2f} dB"
                    )
                    log_payload[f"val/target/{sid}"] = wandb.Image(
                        target_img_r, caption=f"{sid} target @ step {step}"
                    )
                    log_payload[f"val/heatmap/{sid}"] = wandb.Image(
                        heat, caption=f"{sid} heatmap @ step {step}"
                    )
                    log_payload[f"val/psnr/{sid}"] = psnr_val

                if len(psnr_vals) > 0:
                    log_payload["val/psnr_mean"] = float(np.nanmean(psnr_vals))

                if len(log_payload) > 0:
                    accelerator.log(log_payload, step=step)
        finally:
            try:
                model.pipe.scheduler.set_timesteps(1000, training=True)
            except TypeError:
                model.pipe.scheduler.set_timesteps(1000)
        print("after infer timesteps:", len(getattr(model.pipe.scheduler, "timesteps", [])))
        model.train()


    import time
    global_step = 0
    t0 = time.time()
    eval_loss_every = int(getattr(args, "eval_loss_every_steps", 0) or 0)
    eval_infer_every = int(getattr(args, "eval_infer_every_steps", 0) or 0)

    optimizer.zero_grad(set_to_none=True)

    accum_loss_sum = 0.0
    accum_loss_count = 0

    accum_fm_sum = 0.0
    accum_3d_sum = 0.0
    accum_3d_raw_sum = 0.0

    for epoch_id in range(num_epochs):
        for data in tqdm(dataloader):
            with accelerator.accumulate(model):

                if dataset.load_from_cache:
                    out = model({}, inputs=data)
                else:
                    out = model(data)

                if isinstance(out, (tuple, list)) and len(out) == 2 and isinstance(out[1], dict):
                    loss, logs = out
                else:
                    loss, logs = out, {}

                accum_loss_sum += loss.detach().float().item()
                accum_loss_count += 1

                if "loss_main" in logs:
                    accum_fm_sum += logs["loss_main"].detach().float().item()
                if "loss_3d_weighted" in logs:
                    accum_3d_sum += logs["loss_3d_weighted"].detach().float().item()
                if "loss_3d_raw" in logs:
                    accum_3d_raw_sum += logs["loss_3d_raw"].detach().float().item()

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                    mean_loss = accum_loss_sum / max(accum_loss_count, 1)
                    mean_fm = accum_fm_sum / max(accum_loss_count, 1)
                    mean_3d = accum_3d_sum / max(accum_loss_count, 1)
                    mean_3d_raw = accum_3d_raw_sum / max(accum_loss_count, 1)

                    accum_loss_sum = accum_fm_sum = accum_3d_sum = accum_3d_raw_sum = 0.0
                    accum_loss_count = 0

                    model_logger.on_step_end(accelerator, model, save_steps, loss=loss)

                    if accelerator.is_main_process and os.environ.get("WANDB_PROJECT"):
                        lr = optimizer.param_groups[0]["lr"]

                        accelerator.log(
                            {
                                "train/loss": mean_loss,
                                "train/loss_fm": mean_fm,
                                "train/loss_3d": mean_3d,
                                "train/loss_3d_raw": mean_3d_raw,
                                "train/lr": lr,
                                "train/epoch": epoch_id,
                            },
                            step=global_step,
                        )

                        if global_step % 20 == 0 and global_step > 0:
                            dt = time.time() - t0
                            accelerator.log({"train/steps_per_sec": 20.0 / dt}, step=global_step)
                            t0 = time.time()

                    #val loss
                    if eval_loss_every and global_step > 0 and global_step % eval_loss_every == 0:
                        vloss = run_validation_loss()
                        if (vloss is not None) and accelerator.is_main_process and os.environ.get("WANDB_PROJECT"):
                            accelerator.log({"val/loss": vloss}, step=global_step)

                    #val inference
                    if eval_infer_every and global_step > 0 and global_step % eval_infer_every == 0:
                        run_tracked_predictions(global_step)

                    global_step += 1

        if save_steps is None:
            model_logger.on_epoch_end(accelerator, model, epoch_id)

    model_logger.on_training_end(accelerator, model, save_steps)


def launch_data_process_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    num_workers: int = 8,
    args=None,
):
    if args is not None:
        num_workers = args.dataset_num_workers

    dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, collate_fn=lambda x: x[0], num_workers=num_workers)

    model.to(device=accelerator.device)
    model, dataloader = accelerator.prepare(model, dataloader)

    for data_id, data in enumerate(tqdm(dataloader)):
        with accelerator.accumulate(model):
            with torch.no_grad():
                folder = os.path.join(model_logger.output_path, str(accelerator.process_index))
                os.makedirs(folder, exist_ok=True)
                save_path = os.path.join(model_logger.output_path, str(accelerator.process_index), f"{data_id}.pth")
                data = model(data)
                torch.save(data, save_path)