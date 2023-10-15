#!/bin/bash -l 
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB 
#SBATCH --time=24:00:00
#SBATCH --partition=batch
#SBATCH --job-name deeplab_cnet
#SBATCH --output logs/deeplab_cnet_%J.out
#SBATCH --error logs/deeplab_cnet_%J.err
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
python deeplabv3_controlnet.py \
 --deeplab_ckpt deeplab/wandb/run-20231004_185133-u2xfwaqc/files/weights/best.pth \
 --cnet_ckpt /ibex/user/stahlg/wandb/run-20231010_222724-keahqeey/files/weights \
 --condition_type SAM \
 --deeplab_mode eval \
 --dataset_path $WEKAIO/datasets/cityscapes \
 --lr 1e-5 \
 --epochs 400 \
 --batch_size 2 \
 --nworkers 1 \
 --max_timesteps 150 \
 --denoise PARTIAL_DENOISE_T_FIXED \
 --device cuda:0 \
 --exp lr_1e-5_weighted_cnet_sam_finetune_sam_deeplabmode_eval_batch2_epochs400_partial_denoise_T_fixed_nullprompt_normgta_samples3k