#!/bin/bash
#SBATCH -p performance
#SBATCH --job-name=vton_gsplat
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

cd "${SLURM_SUBMIT_DIR:-$PWD}"

module load anaconda

source /opt/conda/etc/profile.d/conda.sh

conda activate gsplat310vton

echo "=== SLURM / GPU DEBUG ==="
echo "JobID=$SLURM_JOB_ID Node=$(hostname)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "NVIDIA_VISIBLE_DEVICES=$NVIDIA_VISIBLE_DEVICES"
nvidia-smi || true
which nvidia-smi || true

cd gsplat/examples

python simple_viewer.py --ckpt ../../data/train/can/can_30/results/qwen_gsplat/ckpts/ckpt_3499_rank0.pt --port 8080

echo "=== Job finished successfully ==="
