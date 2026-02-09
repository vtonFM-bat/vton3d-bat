#!/usr/bin/env bash
#SBATCH -p performance
#SBATCH --job-name=env_setup
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=10G



set -e

echo ">>> Initializing conda"
eval "$(conda shell.bash hook)"



# vton Env

ENV_NAME_VTON=vton

if conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME_VTON"; then
    echo ">>> Conda env '$ENV_NAME_VTON' already exists – skipping creation"
else
    echo ">>> Creating conda env: $ENV_NAME_VTON"
    conda create -n $ENV_NAME_VTON python=3.11 -y
fi

echo ">>> Activating $ENV_NAME_VTON"
conda activate $ENV_NAME_VTON

echo ">>> Python:"
which python
python -V

echo ">>> Upgrading pip"
pip install -U pip

echo ">>> Installing PyTorch CUDA for vton"
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

echo ">>> Installing vton project with pip -e"
pip install -e .

echo ">>> VTON environment done"
conda deactivate
