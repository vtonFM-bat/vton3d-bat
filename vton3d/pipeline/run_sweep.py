# vton3d/pipeline/run_sweep.py
from __future__ import annotations
from pathlib import Path
import argparse
import yaml
import os
import shutil
from copy import deepcopy
from typing import Any
import wandb

from vton3d.pipeline.run_pipeline import load_config, parse_cli_args, run_pipeline


KNOWN_SECTIONS = {"paths", "pipeline", "frame_extraction", "vggt", "qwen", "gsplat", "wandb"}


def _set_by_path(d: dict, path: list[str], value: Any):
    cur = d
    for k in path[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[path[-1]] = value


def apply_dot_overrides(cfg: dict, wb_config: dict) -> dict:
    """
    Takes wandb.config keys like:
      vggt.keep_top_percent=0.3
      pipeline.steps=["vggt"]
    and writes into cfg nested.
    """
    out = deepcopy(cfg)

    for k, v in wb_config.items():
        if not isinstance(k, str):
            continue
        if "." not in k:
            continue
        parts = k.split(".")
        if parts[0] not in KNOWN_SECTIONS:
            continue
        _set_by_path(out, parts, v)

    return out


def prepare_workdir(cfg: dict, run: wandb.sdk.wandb_run.Run) -> Path:
    base = Path(cfg["paths"]["scene_dir"]).expanduser()
    num_frames = cfg.get("extract_frames", {}).get("num_frames", None)
    if num_frames:
        scene_dir = (base / f"{base.name}_{num_frames}").resolve()
    else:
        scene_dir = base.resolve()

    # default work-root: <scene_dir>/runs/<run_name>/<wandb_id>
    run_name = cfg["wandb"].get("run_name", "run")
    work_root = Path(cfg["paths"].get("runs_root", scene_dir / "sweep_runs")).expanduser().resolve()
    work_dir = work_root / run_name / run.id
    work_dir.mkdir(parents=True, exist_ok=True)

    # mirror input images
    mirror = cfg.get("pipeline", {}).get("input_mirror", "symlink")
    src = scene_dir / "real" / "images"
    src.mkdir(parents=True, exist_ok=True)
    dst = work_dir / "real" / "images"
    dst.parent.mkdir(parents=True, exist_ok=True)

    if not dst.exists():
        if mirror == "symlink":
            os.symlink(src, dst, target_is_directory=True)
        elif mirror == "copy":
            shutil.copytree(src, dst)
        else:
            raise ValueError(f"Unknown pipeline.input_mirror: {mirror}")

    return work_dir


def main():
    args = parse_cli_args()
    base_cfg = load_config(args.config)

    # sweep-friendly init:
    # -> W&B sets sweep params in wandb.config (like vggt.keep_top_percent)
    if os.environ.get("WANDB_API_KEY"):
        wandb.login(key=os.environ["WANDB_API_KEY"], relogin=False)

    wb = base_cfg.get("wandb", {})
    run = wandb.init(
        project=wb.get("project", "vton_pipeline"),
        entity=wb.get("entity", None),
        mode=wb.get("mode", "online"),
        config=base_cfg,  # base config wird geloggt
    )

    # merge sweep overrides in nested cfg
    cfg = apply_dot_overrides(base_cfg, dict(run.config))

    # store resolved config in wandb
    run.config.update(cfg, allow_val_change=True)

    # prepare isolated work_dir
    work_dir = prepare_workdir(cfg, run)

    run_pipeline(cfg, work_dir)

    wandb.finish()


if __name__ == "__main__":
    main()
