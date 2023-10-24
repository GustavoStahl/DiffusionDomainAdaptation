# core libraries
import os
import random

# computer vision and arrays
import cv2
import numpy as np

# deep learning
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Normalize

# custom scripts
from controlnet_augmentation import ControlNetAugmentation
from t2i_augmentation import T2IAdapterAugmentation
from base_augmentation import SchedulerType, DenoiseType

# visualization
import argparse
from tqdm import tqdm

from dataclasses import dataclass

from deeplab.kitti_dataset import get_dataset, ConditionType

device = None

@dataclass
class Config(object):
    #NOTE hardcoded prompt
    PROMPT = ""
    # normalization values
    CNET_MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32) # convert [0,1]➙[-1,1]
    CNET_STD  = np.array([0.5, 0.5, 0.5], dtype=np.float32)

def set_determinism():
    # set seed, to be deterministic
    seed = 123
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def generate_aug(cnet_aug, test_dataloader):
    
    prompt = Config.PROMPT
    cnet_normalize = Normalize(Config.CNET_MEAN, Config.CNET_STD)
    
    # rgb_folder = "leftImg8bit"
    rgb_folder = "image_2"
    # aug_folder = "leftImg8bit_aug_lineart"
    aug_folder = "image_2_aug_lineart"

    for data in tqdm(test_dataloader):
        
        image = data["image"]
        filepath = data["filepath"]
        condition = data["condition"]
                                                    
        condition = (condition.float() / 255)
        condition = torch.repeat_interleave(condition, 3, dim=1) # (B, 1, H, W) -> (B, 3, H, W)
                                
        image = image.to(device)
        condition = condition.to(device)
        
        # normalize [0,1]➙[-1,1]
        image_norm = cnet_normalize(image)
                                            
        #NOTE: this expects RGB
        with torch.no_grad():
            diffusion_pred, *_ = cnet_aug(image_norm, condition, prompt)

        diffusion_pred_color = diffusion_pred.detach().permute(0, 2, 3, 1).cpu().numpy()
        diffusion_pred_color = (diffusion_pred_color * 255).astype("uint8")
        diffusion_pred_color = diffusion_pred_color[...,::-1] # RGB➙BGR
        
        for path, sample in zip(filepath, diffusion_pred_color):
            dirname = os.path.dirname(path)
            filename = os.path.basename(path)
            aug_dirname = dirname.replace(rgb_folder, aug_folder)
            os.makedirs(aug_dirname, exist_ok=True)
            cv2.imwrite(os.path.join(aug_dirname, filename), sample)

def main(args):          
    global device
    device = args.device
    
    # avoid randomicity
    set_determinism()
        
    mean_and_std = ((0., 0., 0.), (1., 1., 1.)) # Set normalization in range (0,1)
    condition_type = ConditionType[args.condition_type]
    test_dataset = get_dataset(args.dataset_path, 
                               split=args.dataset_split, 
                               mean_and_std=mean_and_std,
                               condition_type=condition_type)
    test_dataloader = DataLoader(test_dataset, 
                                 batch_size=args.batch_size, 
                                 shuffle=True,
                                 num_workers=args.nworkers,
                                 drop_last=True)
    
    scheduler_type = SchedulerType[args.scheduler]
    denoise_type = DenoiseType[args.denoise]
        
    augmentation_model = ControlNetAugmentation(args.cnet_ckpt, 
                                                scheduler_type=scheduler_type,
                                                eta=args.eta,
                                                max_timesteps=args.max_timesteps,
                                                num_timesteps=args.num_timesteps,
                                                denoise_type=denoise_type,
                                                device=device)
    # augmentation_model = T2IAdapterAugmentation(args.cnet_ckpt, 
    #                                             scheduler_type=scheduler_type,
    #                                             eta=args.eta,
    #                                             max_timesteps=args.max_timesteps,
    #                                             num_timesteps=args.num_timesteps,
    #                                             device=device)
    augmentation_model.set_eval()
           
    # Enable TF32 for faster training on Ampere GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    
    generate_aug(augmentation_model,
                 test_dataloader)
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", 
                        required=True,
                        type=str,
                        help="Path to Cityscapes dataset.")
    parser.add_argument("--dataset_split", 
                        default="val",
                        type=str,
                        help="Which split to use.")      
    parser.add_argument("--batch_size", 
                        type=int,
                        default=8, 
                        help="Number of samples in batch.")        
    parser.add_argument("--nworkers", 
                        type=int,
                        default=0, 
                        help="Number of workers for the dataloader.")       
    parser.add_argument("--condition_type", 
                        type=str,
                        default=ConditionType.NONE.name,
                        choices=ConditionType.options(), 
                        help="Which condition to use in ControlNet.")                        
    parser.add_argument("--cnet_ckpt", 
                        default="lllyasviel/control_v11p_sd15_canny", 
                        help="ControlNet cpkt path.")   
    parser.add_argument("--max_timesteps", 
                        type=int,
                        default=1000, 
                        help="Max number of timesteps in diffusion.")   
    parser.add_argument("--num_timesteps", 
                        type=int,
                        default=20, 
                        help="How many timesteps to use, "
                             "timesteps = max_timesteps / num_timesteps.")   
    parser.add_argument("--scheduler", 
                        type=str,
                        default=SchedulerType.DDIM.name,
                        choices=SchedulerType.options(), 
                        help="Scheduler to be used.")
    parser.add_argument("--eta", 
                        default=0.0,
                        type=float,
                        help="ETA for DDIM.")
    parser.add_argument("--denoise", 
                        type=str,
                        default=DenoiseType.PARTIAL_DENOISE_T_FIXED.name,
                        choices=DenoiseType.options(), 
                        help="Type of denoise to be used.")      
    parser.add_argument("--device", 
                        type=str,
                        default="cuda:0", 
                        help="Which device to use for the training.")       
    return parser.parse_args()
    
if __name__ == "__main__":
    args = parse_args()
    main(args)
    

    