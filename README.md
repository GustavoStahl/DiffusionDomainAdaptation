# Enhancing Image Domain Adaptation with Layout-to-Image Models

This repository contains the implementation of our method for improving domain adaptation in semantic segmentation tasks. By leveraging Layout-to-Image (L2I) models, we bridge the gap between synthetic and real-world domains, enabling more accurate segmentation on real-world datasets.

## Installation

### Prerequisites
* Python 3.8 or higher
* NVIDIA GPUs with CUDA 11.3 or higher

### Environment Setup

Install the required libraries via Conda:

```shell
conda env create -f environment.yml
conda activate <env-name>
```

### Login to wandb (optional)
This project uses Weights & Biases (Wandb) for experiment tracking and visualizing training metrics. To enable Wandb logging, log in with your account:

```shell
# Proceed by entering your API key
wandb login
```

## Usage

### Preparing the datasets

Only the GTA5 dataset requires preprocessing. Use the provided script:

```shell
python deeplab/prepare_gta.py --mat_file_path <gta5-mat-file-path> --input_dir <input-dir-path> --output_dir <output-dir-path>
```

### Training and testing DeepLabv3

Train the segmentation model on synthetic data (e.g., GTA5):

```shell
# Training on synthetic data, change accordingly
python deeplab/train_deeplabv3.py --dataset_path <path-to-gta5-dataset> --dataset_type GTA5 --epochs 100 --batch_size 8 --lr 2.5e-3 --nworkers 16 --filter_labels --device cuda --exp <experiment-name>
```

Evaluate the trained model on real datasets (e.g., Cityscapes):

```shell
python deeplab/test_deeplabv3.py --ckpt_path wandb/<run-name>/files/weights/best.pth --dataset_path <testing-dataset-path> --dataset_type CITYSCAPES --batch_size 6 --nworkers 4 --filter_labels --device cuda:0 --exp <experiment-name>
```

> The DeepLabv3 pre-trained models available in our paper can be found here: pretrained o [GTA5](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/gustavo_stahl_mbzuai_ac_ae/ESt3OtSjEeNCl3DSRGhWstUBnrzUwayaac1FzozoQIEiAg?e=YJ8LnO), pretrained on [Cityscapes](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/gustavo_stahl_mbzuai_ac_ae/Ech4pZrjJOFDu0FDgOdjeQkBzRFiWGP_5ITIUyCoCES0uw?e=YW3Xbl).

### Training and testing Controlnet Augmentation

Fine-tune ControlNet for real-to-synthetic mapping:

```shell
python deeplabv3_controlnet.py --deeplab_ckpt deeplab/wandb/<run-name>/files/weights/best.pth --deeplab_mode eval --dataset_path <real-dataset-path> --condition_type CANNY --lr 1e-4 --epochs 100 --batch_size 8 --nworkers 4 --max_timesteps 150 --denoise PARTIAL_DENOISE_T_FIXED --exp <experiment-name> --device cuda
```

It's important to notice that you are supposed to load the pre-trained DeepLabv3 weights trained on synthetic images. Morevoer, the code support different types of conditioning methods besides Canny, such as depth maps and SAM masks. For these extra methdos, their guidance files should be precomputed and stored within the training set, for specifics check the dataloaders available in `deeplab` folder. At the moment the Dataloader that supports these extra guides is only implemented for Cityscapes and KITTI, but one can easily adapt the code. 

> The pre-trained weights presented in our paper for ControlNet can be found [here](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/gustavo_stahl_mbzuai_ac_ae/EUBhjLWJKURFmxL91Bi0czUBVTATMoPe_g8hx6S0kBR-eA?e=42VegP).

Testing can be easily achieved by running:

```shell
python test_deeplabv3_controlnet.py --cnet_ckpt wandb/<run-path>/files/weights/best/ --deeplab_ckpt deeplab/wandb/<run-name>/files/weights/best.pth --deeplab_mode eval --dataset_path <dataset-path> --condition_type CANNY --batch_size 4 --nworkers 4 --max_timesteps 150 --denoise PARTIAL_DENOISE_T_FIXED --dataset_type CITYSCAPES --filter_labels --device cuda --exp <experiment-name>
```

## Acknowledgments
* The segmentation model is based on DeepLabv3.
* The L2I model leverages ControlNet built on Stable Diffusion v1.5.
* Datasets used: GTA5, Cityscapes, and KITTI.

