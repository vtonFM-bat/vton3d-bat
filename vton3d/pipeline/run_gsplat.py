#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict


def load_config(path: Path) -> Dict[str, Any]:
    """
    Lädt Config aus YAML oder JSON.
    - YAML: benötigt pyyaml (pip install pyyaml)
    - JSON: keine extra deps
    """
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8")

    if suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "YAML config requires PyYAML. Install with: pip install pyyaml"
            ) from e
        cfg = yaml.safe_load(text)
        if not isinstance(cfg, dict):
            raise ValueError("YAML root must be a mapping/dict.")
        return cfg

    if suffix == ".json":
        cfg = json.loads(text)
        if not isinstance(cfg, dict):
            raise ValueError("JSON root must be an object/dict.")
        return cfg

    raise ValueError(f"Unsupported config format: {suffix} (use .yaml/.yml or .json)")


def run_step_gsplat(cfg: dict) -> None:
    print("=== [Step 3] GSplat training ===")

    gs_cfg = cfg.get("gsplat", {})
    test_every = gs_cfg.get("test_every", 8)
    data_factor = gs_cfg.get("data_factor", 1)
    eval_steps = gs_cfg.get("eval_steps", [7000, 30000])
    tile_size = gs_cfg.get("tile_size", 16)
    max_steps = gs_cfg.get("max_steps", 30000)
    disable_video = gs_cfg.get("disable_video", False)
    disable_viewer = gs_cfg.get("disable_viewer", True)

    # Erwartet: cfg["paths"]["scene_dir"]
    project_root = Path(__file__).resolve().parents[2]
    gsplat_repo = project_root / "gsplat"

    base_scene_dir = Path(cfg["paths"]["scene_dir"])

    ef_cfg = cfg.get("extract_frames", {}) or {}
    num_frames = int(ef_cfg.get("num_frames", 0))

    scene_dir = base_scene_dir / f"{base_scene_dir.name}_{num_frames}"

    data_dir = Path("..") / scene_dir / "qwen"
    result_dir = Path("..") / scene_dir / "results" / "qwen_gsplat"

    cmd = [
        sys.executable,
        "examples/gsplat_trainer_pipe.py",
        "default",
        "--data_dir", str(data_dir),
        "--data_factor", str(data_factor),
        "--result_dir", str(result_dir),
        "--test_every", str(test_every),
        "--eval_steps", *[str(s) for s in eval_steps],
        "--tile_size", str(tile_size),
        "--max_steps", str(max_steps),
        "--disable_video", str(disable_video),
        "--disable_viewer", str(disable_viewer),
    ]

    print(f"  -> gsplat repo: {gsplat_repo}")
    print(f"  -> running: {' '.join(cmd)}")

    subprocess.run(cmd, cwd=str(gsplat_repo), check=True)

    print("=== [Step GSplat] Done ===\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="Path to config file (.yaml/.yml or .json)",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config).expanduser().resolve()
    cfg = load_config(cfg_path)

    steps = (cfg.get("pipeline", {}) or {}).get("steps", {}) or {}
    run_gsplat = bool(steps.get("gsplat", False))

    if run_gsplat:
        run_step_gsplat(cfg)
    else:
        print("=== [Step 3] GSplat training skipped (pipeline.steps.gsplat = false) ===")


if __name__ == "__main__":
    main()
