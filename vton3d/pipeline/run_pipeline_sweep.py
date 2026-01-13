# vton3d/pipeline/run_pipeline_sweep.py

from __future__ import annotations
from pathlib import Path
import argparse
import yaml
import torch
import os
import shutil
from copy import deepcopy
from typing import Any
import wandb
from PIL import Image

from vton3d.vggt.run_vggt import vggt2colmap
from vton3d.qwen.run_qwen import run_qwen_from_config_dict
from argparse import Namespace

# -------------------------
# helpers (existing: normalize/resize stay)
# -------------------------

def load_config(config_path: str | Path) -> dict:
    config_path = Path(config_path)
    with config_path.open("r") as f:
        return yaml.safe_load(f)

# -------------------------
# sweep override utilities (extended)
# -------------------------

KNOWN_SECTIONS = {"paths", "pipeline", "vggt", "qwen", "gsplat", "wandb"}

def _deep_merge(dst: dict, src: dict) -> dict:
    """recursive dict merge: src overwrites dst"""
    out = deepcopy(dst)
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out

def _set_by_path(d: dict, path: list[str], value: Any) -> None:
    cur = d
    for key in path[:-1]:
        if key not in cur or not isinstance(cur[key], dict):
            cur[key] = {}
        cur = cur[key]
    cur[path[-1]] = value

def _parse_sweep_key(key: str, known_sections: set[str]) -> list[str] | None:
    # section__a__b
    if "__" in key:
        parts = key.split("__")
        if parts[0] in known_sections and all(parts):
            return parts
    # section.a.b
    if "." in key:
        parts = key.split(".")
        if parts[0] in known_sections and all(parts):
            return parts
    return None

def apply_sweep_overrides(cfg: dict, wb_config) -> dict:
    """
    Supports BOTH:
      - nested wandb.config dicts (recommended sweep style)
      - flattened keys: vggt.keep_top_percent or vggt__keep_top_percent
    """
    new_cfg = deepcopy(cfg)
    flat = dict(wb_config)

    # 1) nested sections
    for sec in KNOWN_SECTIONS:
        if sec in flat and isinstance(flat[sec], dict):
            new_cfg[sec] = _deep_merge(new_cfg.get(sec, {}), flat[sec])

    # 2) flattened keys
    for k, v in flat.items():
        if not isinstance(k, str):
            continue
        path = _parse_sweep_key(k, KNOWN_SECTIONS)
        if path is not None:
            _set_by_path(new_cfg, path, v)

    return new_cfg

# -------------------------
# run-dir / mirroring
# -------------------------

def _resolve_work_dir(cfg: dict, run: wandb.sdk.wandb_run.Run | None) -> Path:
    paths = cfg.setdefault("paths", {})
    scene_dir = Path(paths["scene_dir"]).expanduser().resolve()

    tmpl = paths.get("work_dir")
    if not tmpl:
        return scene_dir

    # simple template replacement
    s = str(tmpl)
    # optional: allow ${run_name}
    s = s.replace("${run_name}", str(cfg.get("run_name", "run")))
    if run is not None:
        s = s.replace("{run_id}", run.id)
    else:
        s = s.replace("{run_id}", "no_wandb")

    return Path(s).expanduser().resolve()

def _ensure_work_tree(cfg: dict, work_dir: Path) -> None:
    """
    Ensure expected structure in work_dir:
      work_dir/real/images (mirrored)
      work_dir/qwen/images (created when needed)
    """
    pipeline_cfg = cfg.get("pipeline", {})
    mirror = pipeline_cfg.get("input_mirror", "symlink")

    scene_dir = Path(cfg["paths"]["scene_dir"]).expanduser().resolve()
    src_images = scene_dir / "real" / "images"
    dst_images = work_dir / "real" / "images"

    if not src_images.exists():
        raise FileNotFoundError(f"Missing source images dir: {src_images}")

    dst_images.parent.mkdir(parents=True, exist_ok=True)

    if dst_images.exists():
        # already prepared
        return

    if mirror == "symlink":
        os.symlink(src_images, dst_images, target_is_directory=True)
    elif mirror == "copy":
        shutil.copytree(src_images, dst_images)
    else:
        raise ValueError(f"Unknown pipeline.input_mirror: {mirror}")

def _get_base_dir(cfg: dict) -> Path:
    """Base dir for pipeline steps (work_dir if configured, else scene_dir)."""
    paths = cfg.get("paths", {})
    return Path(paths.get("work_dir_resolved", paths["scene_dir"])).expanduser().resolve()

# -------------------------
# existing functions you have: build_vggt_args_from_config etc.
# adjust to use work_dir base
# -------------------------

def build_vggt_args_from_config(cfg: dict) -> Namespace:
    base_scene_dir = _get_base_dir(cfg)
    real_scene_dir = base_scene_dir / "real"

    vggt_cfg = cfg.get("vggt", {})
    return Namespace(
        scene_dir=str(real_scene_dir),
        seed=int(vggt_cfg.get("seed", 42)),
        use_ba=bool(vggt_cfg.get("use_ba", False)),
        max_reproj_error=float(vggt_cfg.get("max_reproj_error", 8.0)),
        shared_camera=bool(vggt_cfg.get("shared_camera", False)),
        camera_type=str(vggt_cfg.get("camera_type", "SIMPLE_PINHOLE")),
        vis_thresh=float(vggt_cfg.get("vis_thresh", 0.2)),
        query_frame_num=int(vggt_cfg.get("query_frame_num", 8)),
        max_query_pts=int(vggt_cfg.get("max_query_pts", 4096)),
        fine_tracking=bool(vggt_cfg.get("fine_tracking", True)),
        conf_thres_value=float(vggt_cfg.get("conf_thres_value", 5.0)),
        keep_top_percent=float(vggt_cfg.get("keep_top_percent", 0.2)),
    )

def run_step_vggt_colmap(cfg: dict):
    print("=== [Step 1] VGGT + COLMAP Reconstruction ===")
    vggt_args = build_vggt_args_from_config(cfg)
    print(f"  -> scene_dir (work): {Path(vggt_args.scene_dir).parent}")
    with torch.no_grad():
        vggt2colmap(vggt_args)
    print("=== [Step VGGT] Done ===\n")

def copy_colmap_sparse(real_scene_dir: Path, qwen_scene_dir: Path):
    src = real_scene_dir / "sparse"
    dst = qwen_scene_dir / "sparse"
    if not src.exists():
        raise FileNotFoundError(f"Missing COLMAP sparse folder: {src}")
    shutil.copytree(src, dst, dirs_exist_ok=True)
    print(f"  -> Copied COLMAP sparse: {src}  ->  {dst}")

def run_step_qwen_clothing(cfg: dict):
    print("=== [Step 2] Qwen VTON edit ===")
    base_scene_dir = _get_base_dir(cfg)

    input_dir = base_scene_dir / "real" / "images"
    output_dir = base_scene_dir / "qwen" / "images"
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    qwen_cfg = cfg.get("qwen", {}).copy()
    if not qwen_cfg:
        raise ValueError("Missing 'qwen' section in config for Qwen clothing step.")

    qwen_cfg["source_dir"] = str(input_dir)
    qwen_cfg["output_dir"] = str(output_dir)

    run_qwen_from_config_dict(qwen_cfg)

    real_scene_dir = base_scene_dir / "real"
    qwen_scene_dir = base_scene_dir / "qwen"
    copy_colmap_sparse(real_scene_dir, qwen_scene_dir)

    print("=== [Step Qwen] Done ===\n")

# -------------------------
# main
# -------------------------

def run_pipeline(config_path: str | Path, sweep_overrides: bool = False):
    cfg = load_config(config_path)

    # Cluster-friendly: nicht interaktiv login() – nutze WANDB_API_KEY env.
    # wandb.login() ist ok lokal, aber auf SLURM häufig nervig.
    if os.environ.get("WANDB_API_KEY"):
        wandb.login(key=os.environ["WANDB_API_KEY"], relogin=False)

    wb = cfg.get("wandb", {})
    run = wandb.init(
        project=wb.get("project", "vton_pipeline"),
        entity=wb.get("entity", None),
        name=f"{cfg.get('run_name','run')}",
        tags=wb.get("tags", None),
        mode=wb.get("mode", "online"),
        config=cfg,
        # WICHTIG: für sweeps NICHT mit fixem id/resume arbeiten
        id=None if sweep_overrides else os.environ.get("WANDB_RUN_ID"),
        resume=None if sweep_overrides else os.environ.get("WANDB_RESUME", "allow"),
    )

    # Sweep overrides in cfg zurückspiegeln
    if sweep_overrides and run is not None:
        cfg = apply_sweep_overrides(cfg, run.config)
        run.config.update(cfg, allow_val_change=True)

    # work_dir auflösen und in cfg ablegen
    work_dir = _resolve_work_dir(cfg, run)
    cfg.setdefault("paths", {})["work_dir_resolved"] = str(work_dir)

    _ensure_work_tree(cfg, work_dir)

    # preprocess nur, wenn enabled UND keine symlink-mirror (sonst mutierst du das dataset!)
    pp = cfg.get("pipeline", {}).get("preprocess", {})
    mirror = cfg.get("pipeline", {}).get("input_mirror", "symlink")
    if mirror == "symlink" and (pp.get("normalize_to_png") or pp.get("resize", {}).get("enabled")):
        raise RuntimeError(
            "preprocess ist aktiv, aber input_mirror=symlink. "
            "Das würde dein Dataset in-place verändern. "
            "=> preprocess deaktivieren oder input_mirror=copy nutzen."
        )

    real_images_dir = work_dir / "real" / "images"

    if pp.get("normalize_to_png", False):
        remove_jpg = bool(pp.get("remove_jpg", False))
        normalize_images_to_png(real_images_dir, remove_jpg=remove_jpg)

    resize_cfg = pp.get("resize", {})
    if resize_cfg.get("enabled", False):
        resize_images_to_exact_size(
            real_images_dir,
            target_height=int(resize_cfg.get("target_height", 1248)),
            target_width=int(resize_cfg.get("target_width", 704)),
        )

    # steps
    steps = cfg.get("pipeline", {}).get("steps", ["vggt", "qwen"])
    steps = list(steps)  # ensure list
    print(f"[Pipeline] steps={steps}")
    print(f"[Pipeline] scene_dir={cfg['paths']['scene_dir']}")
    print(f"[Pipeline] work_dir={work_dir}")

    if "vggt" in steps:
        run_step_vggt_colmap(cfg)

    if "qwen" in steps:
        run_step_qwen_clothing(cfg)

    print("[Pipeline] All defined steps completed.")
    wandb.finish()

def parse_cli_args():
    parser = argparse.ArgumentParser(description="VTON3D Pipeline Runner")
    parser.add_argument("--config", type=str, required=True, help="Path to a YAML config file")
    parser.add_argument("--sweep", action="store_true", help="Apply W&B sweep overrides")
    return parser.parse_args()

if __name__ == "__main__":
    cli_args = parse_cli_args()
    run_pipeline(cli_args.config, sweep_overrides=cli_args.sweep)
