# vton3d-bat
Main Repo for Virtual Try-On Pipeline

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

3. Create a folder `data/videos` and put your input video/s there.
4. run python scripts/extract_frames.py -n `numberofframes` (for further params see extract_frames.py)
5. go into configs/vton_pipeline.yaml and set your desired parameters 
(at least the input folder where the images of the person are under: path: scene_dir, 
and the clothing image under: qwen: clothing_image)
6. run sbatch run_pipeline.sh to run the pipeline on your images
```
For Scicore Cluster with a100 and a100-80g GPUs:

```
Run the following command to create the envs:
bash scicore_create_envs.sh


If you want to use the Pipeline with the full Precision Qwen Image Edit Model, change the model 
in the configs to: Qwen/Qwen-Image-Edit-2511
For the full precision model you need at least 
50GB GPU memory (a100-80g) and about 65GB CPU Memory.
For the 4-bit quantized model (ovedrive/Qwen-Image-Edit-2511-4bit) you need 
at least 20GB GPU memory (a100 which has 40GB) or SLURM FHNW with A4500 GPU.

Then run the pipeline with:
sbatch scicore_run_pipeline.sh

```

# Create Sweep for eperiments with WandB

1. Slurm-Terminal
   - wandb sweep configs/sweeps/vggt_sweep.yaml (copy outputpath sweepid)
2. Run Sweep with:
   - sbatch run_pipeline_wandb_sweep.sh team_entity/vton_pipeline/sweepid
