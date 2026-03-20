# Combining 2D Flow Matching and Gaussian Splatting for 3D Virtual Try-on

This repository contains the code, experiments, and supplementary material for our bachelor's thesis on combining 2D flow matching and Gaussian splatting for 3D virtual try-on.


## Main usage
```
Installation (Only Linux) works on FHNW SLURM Cluster:
1. Go in create_envs.sh and set your desired Cuda Pytorch version 
(from https://pytorch.org/get-started/locally/) in this line: 

echo ">>> Installing PyTorch CUDA for vton"
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

2. Create Conda Envs by running sbatch create_envs.sh 
This will create two Conda Envs. 
One for the VTON and VGGT model (vton) and one for the Gaussian Splatting Model (gsplat310vton) 
as they both need same libraries in different versions.

3. Create a folder `data/video` and put your input video/s there.
4. run python scripts/extract_frames.py -n `numberofframes` (for further params see extract_frames.py)
5. go into configs/vton_pipeline.yaml and set your desired parameters 
(at least the input folder where the images of the person are under: path: scene_dir, 
and the clothing image under: qwen: clothing_image)
6. run sbatch run_pipeline.sh to run the pipeline on your images
7. view the results with:
 - sbatch view_gsplat.sh (change the checkpoint path in the script to your path: scene_dir / person_framenumber /results/qwen_gsplat/ckpts/ckpt_epoch_XXX.pt)
 - ssh hostname -N -L port:nodename:port username@hostname (ssh slurm -N -L 8080:calc-g-008:8080 example.user@slurm)
 - open your browser and go to http://localhost:viwer to view the results in the viewer.
```
Create Sweep for experiments with WandB on Slurm Cluster
```
1. activate env
    - conda activate vton 
2. Slurm-Terminal
   - wandb sweep configs/sweeps/vggt_sweep.yaml (copy outputpath sweepid)
3. Run Sweep with:
   - sbatch run_pipeline_wandb_sweep.sh team_entity/vton_pipeline/sweepid

```
## Acess and using SciCore Cluster (just for users with Scicore Account)

For Scicore Cluster with a100 and a100-80g GPUs:
```
Run the following command to create the envs:
bash scicore_create_envs.sh


1. If you want to use the Pipeline with the full Precision Qwen Image Edit Model, change the model in the configs to: Qwen/Qwen-Image-Edit-2511
- For the full precision model you need at least 50GB GPU memory (a100-80g) and about 65GB CPU Memory.
- For the 4-bit quantized model (ovedrive/Qwen-Image-Edit-2511-4bit) you need at least 20GB GPU memory (a100 which has 40GB) or SLURM FHNW with A4500 GPU.

2. Then run the pipeline with:
3. sbatch scicore_run_pipeline.sh
```
View checkpoints from Gaussian Splatting (Viser) from Scicore Cluster
```
1. connect to scicore
2. open scicore_view_gsplat.sh
    - change --ckpt param to your desired checkpoint path
3. run the script with
    - sbatch scicore_view_gsplat.sh
4. open a local terminal
    - ssh -N -L 8080:YourScicoreNodename:8080 YourUsername@login12.scicore.unibas.ch
5. open your browser and go to http://localhost:8080
```

## Repository Structure

Quick overview of the folders and files

- `configs/`
  - `vton_pipeline.yaml` — main configuration for the pipeline (all needed paramters for VGGT, Qwen, OpticalFlow, GSplat, etc.).
  - `sweeps/` — sweep configs (example: `vggt_sweep.yaml`).
- `data/`
  - `clothing/` — reference clothing images (organized by category).
    - `train/dress/`, — dress image.
    - `train/upper/short/shirts` — shirt images. 
    - `train/lower/long/pants` — pant images.
  - `train/` — training data
    - `person/` — folder's with person images (extracted from videos with number of used frames as suffix). 
      - `video/` — folder with the video of the person. 
  - `val/`— test data
    - `person/` — folder's with person images (extracted from videos with number of used frames as suffix). 
      - `video/` — folder with the video of the person. 
  - `other/` — other resources.

- `gsplat/` — Gaussian Splatting code and examples (viewer/trainer and dependencies).

- `lora_train/` — models and helpers for LoRA training; contains references to QWEN models and example scripts.

- `models/` — pre-trained models (e.g. QWEN, VGGT) and checkpoints.

- `notebooks/` — Jupyter notebooks for every experiment with visualizations.

- `Sapiens-Pytorch-Inference/` — code used for segmentation.

- `scripts/` — scripts.

- `vton3d/` — main Python package for the pipeline with subpackages.
  - `pipeline/` — main pipeline code.
    - `run_pipeline.py` — main script to run the pipeline.
    - `run_gsplat.py` — script to run Gaussian Splatting.
    - `run_sweep.py` — script to run wandb sweeps.
  - `qwen/` — code related to run QWEN.
  - `vggt/` — code related to run VGGT.
  - `utils/` — utility functions 
    - `background_segmentation.py`,
    - `depth_maps.py`,
    - `extract_frames.py`,
    - `masked_optical_flow.py`,
    - `qwen_eval.py`,
    - `utils.py`,
    - `vggt_eval.py`

- **`create_envs.sh` — script to create needed conda environments.**
- `create_train_envs.sh` — script to create conda environments for LoRA training.
- `create_vton_env.sh` — script to create a conda environment just for the VTON section of the pipeline.
- `run_experiments.sh` — script to run specific experiments in the pipeline.
- `run_exps_pipeline.sh` — script to run the pipeline with a specific config (for the experiments).
- **`run_pipeline.sh` — main script to run the pipeline.**
- `run_pipeline_wandb_sweep.sh` — script to run a wandb sweep.
- `scicore_create_envs.sh` — script to create conda envs on Scicore Cluster.
- `scicore_run_pipeline.sh` — script to run the pipeline on Scicore Cluster.
- `scicore_view_gsplat.sh` — script to view Gaussian Splatting checkpoints on Scicore Cluster.
- `train_qwen.sh` — script to train QWEN LoRA models.
- **`view_gsplat.sh` — script to view Gaussian Splatting checkpoints on Slur Calculon Cluster.**

## FAQ

**Where is the QWEN prompt defined?**

- The QWEN prompts (positive and negative) are defined in `configs/vton_pipeline.yaml` under the `qwen:` section. There are `prompts:` and `negative_prompts:` keys with sub-keys `upper`, `lower`, and `dress`.
  - Example: `qwen.prompts.upper` holds the positive prompt for upper garments.
  - This allows separate prompts for each clothing type (`upper`, `lower`, `dress`).

