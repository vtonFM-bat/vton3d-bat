#!/bin/bash
#SBATCH -p p4500
#SBATCH --job-name=wandb_agent_vggt
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=46G
#SBATCH --array=1-20

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
module load anaconda
source /opt/conda/etc/profile.d/conda.sh
conda activate vton

mkdir -p logs

# WICHTIG: NICHT setzen:
unset WANDB_RUN_ID
unset WANDB_RESUME

SWEEP_ID="$1"  # z.B. "dein_entity/vton_pipeline/abc123xy"
wandb agent --count 1 "$SWEEP_ID"
