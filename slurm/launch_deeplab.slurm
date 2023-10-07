#!/bin/bash -l 
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB 
#SBATCH --time=24:00:00
#SBATCH --partition=batch
#SBATCH --job-name deeplab
#SBATCH --output logs/deeplab_%J.out
#SBATCH --error logs/deeplab_%J.err
#SBATCH --mail-user=gustavo.stahl@kaust.edu.sa
#SBATCH --mail-type=ALL
#SBATCH --reservation=A100
#SBATCH --gres=gpu:a100:1

export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

module add cuda/11.0.1
source /home/stahlg/.bashrc
conda activate syn2real

cd /home/stahlg/syn2real_sd/deeplab
python train_deeplabv3.py --dataset_path /ibex/user/stahlg/datasets/gta/prepared_dataset --dataset_type GTA5 --batch_size 16 --filter_labels --device cuda --lr 2.5e-3 --epochs 200 --nworkers 4 --exp gta_weighted_novoid_filterlabels_before_resize_SGD_lr2.5e-3_epochs200
# python train_deeplabv3.py --dataset_path /ibex/user/stahlg/datasets/cityscapes/ --dataset_type CITYSCAPES --batch_size 16 --device cuda --lr 2.5e-3 --epochs 200 --nworkers 4 --exp cityscapes_aug_SGD_lr2.5e-3_epochs200
