#!/bin/bash
#SBATCH --job-name=vton_pipeline
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=10G
#SBATCH --time=01:00:00
#SBATCH --qos=gpu6hours
#SBATCH --output=logs/vton_pipeline.o%j
#SBATCH --error=logs/vton_pipeline.e%j
#SBATCH --partition=a100-80g
#SBATCH --gres=gpu:1

set -euo pipefail

module purge || true
module load Miniconda3
eval "$(conda shell.bash hook)"

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

CONFIG_PATH=${1:-configs/vton_pipeline.yaml}

# ------------------------------------------------------------
# Sweep Values
# ------------------------------------------------------------
test_num_frames=(1 2)

# ------------------------------------------------------------
# Backup config (safety)
# ------------------------------------------------------------
TS="$(date +%Y%m%d_%H%M%S)"
CONFIG_BAK="${CONFIG_PATH}.bak_${TS}"
cp "$CONFIG_PATH" "$CONFIG_BAK"

echo "=== Config backup created ==="
echo "CONFIG_PATH = $CONFIG_PATH"
echo "BACKUP      = $CONFIG_BAK"
echo

# ------------------------------------------------------------
# YAML updater (tries ruamel.yaml first, then pyyaml)
# It searches recursively for extract_frames.num_frames.
# If not found, it creates extract_frames at root.
# ------------------------------------------------------------
update_num_frames_in_yaml() {
  local cfg="$1"
  local nf="$2"

  python - "$cfg" "$nf" <<'PY'
import sys
from pathlib import Path

cfg_path = sys.argv[1]
val = int(sys.argv[2])

text = Path(cfg_path).read_text()

def set_num_frames_anywhere(obj, val):
    """
    Recursively search for a dict key 'extract_frames' with a dict value,
    then set ['num_frames'] = val.
    Returns True if updated somewhere, else False.
    """
    if isinstance(obj, dict):
        if "extract_frames" in obj and isinstance(obj["extract_frames"], dict):
            obj["extract_frames"]["num_frames"] = val
            return True
        for k, v in obj.items():
            if set_num_frames_anywhere(v, val):
                return True
    elif isinstance(obj, list):
        for item in obj:
            if set_num_frames_anywhere(item, val):
                return True
    return False

# ---- Try ruamel.yaml (preserves comments/format better)
try:
    from ruamel.yaml import YAML
    yaml = YAML()
    yaml.preserve_quotes = True
    data = yaml.load(text) or {}

    updated = set_num_frames_anywhere(data, val)
    if not updated:
        # create at root if not found
        if "extract_frames" not in data or data["extract_frames"] is None:
            data["extract_frames"] = {}
        if not isinstance(data["extract_frames"], dict):
            raise TypeError("Found extract_frames but it is not a dict; cannot set num_frames safely.")
        data["extract_frames"]["num_frames"] = val

    with open(cfg_path, "w") as f:
        yaml.dump(data, f)

    print(f"[YAML] Updated extract_frames.num_frames -> {val} (ruamel.yaml)")
    sys.exit(0)

except ImportError:
    pass

# ---- Fallback PyYAML
try:
    import yaml as pyyaml
    data = pyyaml.safe_load(text) or {}

    updated = set_num_frames_anywhere(data, val)
    if not updated:
        if "extract_frames" not in data or data["extract_frames"] is None:
            data["extract_frames"] = {}
        if not isinstance(data["extract_frames"], dict):
            raise TypeError("Found extract_frames but it is not a dict; cannot set num_frames safely.")
        data["extract_frames"]["num_frames"] = val

    Path(cfg_path).write_text(
        pyyaml.safe_dump(data, sort_keys=False, default_flow_style=False)
    )

    print(f"[YAML] Updated extract_frames.num_frames -> {val} (PyYAML)")
    sys.exit(0)

except Exception as e:
    print(f"[YAML] ERROR: could not update YAML safely: {e}", file=sys.stderr)
    sys.exit(1)
PY
}

# ------------------------------------------------------------
# Run helpers
# ------------------------------------------------------------
slurm_gpu_debug() {
  echo "=== SLURM / GPU DEBUG ==="
  echo "JobID=${SLURM_JOB_ID:-NA} Node=$(hostname)"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-NA}"
  echo "NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-NA}"
  nvidia-smi || true
  which nvidia-smi || true
  echo
}

run_full_pipeline() {
  local cfg="$1"
  local nf="$2"

  # separate log files per run
  local log_pipe="logs/run_pipeline_nf${nf}.log"
  local log_gs="logs/run_gsplat_nf${nf}.log"

  echo ">>> [RUN] num_frames=${nf}"
  echo ">>> Config: ${cfg}"
  echo

  # vton env: pipeline
  conda activate vton
  slurm_gpu_debug
  python -m vton3d.pipeline.run_pipeline --config "$cfg" 2>&1 | tee "$log_pipe"
  conda deactivate

  # gsplat env: gsplat
  conda activate gsplat310vton
  slurm_gpu_debug
  python -m vton3d.pipeline.run_gsplat --config "$cfg" 2>&1 | tee "$log_gs"
  conda deactivate

  echo ">>> [DONE] num_frames=${nf}"
  echo
}

# ------------------------------------------------------------
# Main sweep loop
# ------------------------------------------------------------
echo "=== Starting num_frames sweep ==="
echo "Frames list: ${test_num_frames[*]}"
echo

for nf in "${test_num_frames[@]}"; do
  echo "=============================="
  echo "Setting extract_frames.num_frames = ${nf}"
  update_num_frames_in_yaml "$CONFIG_PATH" "$nf"

  # optional: quick preview
  echo "--- YAML preview (extract_frames block) ---"
  grep -n "extract_frames" -A6 "$CONFIG_PATH" || true
  echo

  # set per-run wandb IDs (so they don't overwrite each other)
  export WANDB_RUN_ID="slurm-${SLURM_JOB_ID:-manual}-nf${nf}"
  export WANDB_NAME="vton_pipeline_nf${nf}"

  run_full_pipeline "$CONFIG_PATH" "$nf"
done

echo "=== ALL RUNS FINISHED SUCCESSFULLY ==="
echo "Backup of original config: $CONFIG_BAK"
