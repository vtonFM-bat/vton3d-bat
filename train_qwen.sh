#!/bin/bash
#SBATCH -p h200
#SBATCH --job-name=vton_normal_lora_qwen
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=2-02:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=105G

set -euo pipefail

echo ">>> Starting SLURM job"
echo "Host: $(hostname)"
echo "CUDA visible devices: $CUDA_VISIBLE_DEVICES"

# -------------------------------------------------
# 1) Ins Submit-Verzeichnis gehen (Repo-Root)
# -------------------------------------------------
cd "$SLURM_SUBMIT_DIR"

# -------------------------------------------------
# 2) Conda aktivieren
# -------------------------------------------------
eval "$(conda shell.bash hook)"
conda activate vton_train_qwen

echo ">>> Python:"
which python
python -V

echo ">>> Torch check:"
python -c "import torch; print('Torch', torch.__version__, 'CUDA', torch.version.cuda, 'Available', torch.cuda.is_available())"


export WANDB_PROJECT="qwen_train_lora"
export WANDB_RUN_NAME="normal_lora"
export WANDB_ENTITY="vton_pipeline"
export WANDB_MODE=online


# -------------------------------------------------
# 3) In DiffSynth Repo wechseln
# -------------------------------------------------
cd lora_train/DiffSynth-Studio

# -------------------------------------------------
# 4) Training starten
# -------------------------------------------------

python -c "import os; print('WANDB_PROJECT', os.environ.get('WANDB_PROJECT')); print('WANDB_ENTITY', os.environ.get('WANDB_ENTITY')); print('WANDB_RUN_NAME', os.environ.get('WANDB_RUN_NAME'))"
python -c "import wandb; print('wandb import ok', wandb.__version__)"
python -c "import wandb; import os; print('netrc exists', os.path.exists(os.path.expanduser('~/.netrc')))"

accelerate launch examples/qwen_image/model_training/train.py \
  --dataset_base_path ../data/train \
  --dataset_metadata_path ../data/train/metadata.json \
  --data_file_keys "image,edit_image" \
  --extra_inputs "edit_image" \
  --max_pixels 1048576 \
  --dataset_repeat 1 \
  --num_epochs 16 \
  --gradient_accumulation_steps 2 \
  --learning_rate 1e-4 \
  --model_id_with_origin_paths "Qwen/Qwen-Image-Edit-2511:transformer/diffusion_pytorch_model*.safetensors,Qwen/Qwen-Image:text_encoder/model*.safetensors,Qwen/Qwen-Image:vae/diffusion_pytorch_model.safetensors" \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path ../models/qwen_normal_lora \
  --lora_base_model dit \
  --lora_target_modules "to_q,to_k,to_v,add_q_proj,add_k_proj,add_v_proj,to_out.0,to_add_out,img_mlp.net.2,img_mod.1,txt_mlp.net.2,txt_mod.1" \
  --lora_rank 32 \
  --lora_dropout 0.2 \
  --use_gradient_checkpointing \
  --dataset_num_workers 8 \
  --find_unused_parameters \
  --zero_cond_t \
  --val_dataset_base_path ../data/val \
  --val_dataset_metadata_path ../data/val/metadata.json \
  --eval_loss_every_steps 200 \
  --eval_infer_every_steps 200 \
  --eval_num_samples 3 \
  --eval_infer_steps 20 \
  --eval_cfg_scale 1.0 \
  --eval_seed 0 \
  --eval_sample_ids "pair_000000_0000,pair_000001_middle,pantpair_000006_prev" \
  --three_d_qwen_layers 48 \
  --three_d_loss_weight 0.01

echo ">>> Training finished"

