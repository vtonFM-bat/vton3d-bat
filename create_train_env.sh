
#!/usr/bin/env bash
#SBATCH -p performance
#SBATCH --job-name=env_qwen_setup
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G

set -euo pipefail

eval "$(conda shell.bash hook)"

ENV_NAME=vton_train_qwen

if conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
  echo ">>> Conda env '$ENV_NAME' already exists – skipping creation"
else
  echo ">>> Creating conda env: $ENV_NAME"
  conda create -n "$ENV_NAME" python=3.11 -y
fi

conda activate "$ENV_NAME"
pip install -U pip wheel setuptools

# Torch explizit (wie bei dir)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# >>> HIER der wichtige Teil: in diffsynth_repo wechseln <<<
cd lora_train/DiffSynth-Studio

# DiffSynth installieren (pyproject.toml wird verwendet)
pip install -e .
pip install matplotlib
python -c "import diffsynth, torch; print('diffsynth ok', torch.__version__, torch.version.cuda, torch.cuda.is_available())"

conda deactivate
