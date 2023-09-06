#!/bin/bash -l 
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB 
#SBATCH --time=8:00:00
#SBATCH --partition=batch
#SBATCH --job-name syn2real
#SBATCH --output slurm.out
#SBATCH --error slurm.err
#SBATCH --mail-user=gustavo.stahl@kaust.edu.sa
#SBATCH --mail-type=ALL
#SBATCH --reservation=A100
#SBATCH --gres=gpu:a100:1

export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

module add cuda/11.0.1
source /home/stahlg/.bashrc
conda activate syn2real

cd /home/stahlg/syn2real_sd
PYTHONPATH=$PYTHONPATH:$PWD/beit2 python beit2_controlnet.py --config beit2/configs/gustavo_controlnet.py --beit_ckpt /ibex/user/stahlg/beit2_training_results/beit2_gustavo_sign_1024/best.pth --cnet_ckpt lllyasviel/control_v11p_sd15_canny --exp lr_1emin5_infer_steps_50_samples1k