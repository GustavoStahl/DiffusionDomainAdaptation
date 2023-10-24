from glob import glob
import os

import cv2
from PIL import Image
import numpy as np
from tqdm import tqdm

from controlnet_aux import MidasDetector, LineartDetector

if __name__ == "__main__":
    device = "cuda"
    # dataset_path = "/ibex/user/stahlg/datasets/cityscapes"
    dataset_path = "/ibex/user/stahlg/datasets/kitti_semantic"
    
    # rgb_folder = "leftImg8bit"
    rgb_folder = "image_2"
    # processor_folder = "leftImg8bit_lineart"
    processor_folder = "depth"
    
    processor = MidasDetector.from_pretrained("lllyasviel/Annotators").to(device)
    # processor = LineartDetector.from_pretrained("lllyasviel/Annotators").to(device)

    for split in ["training"]:
    # for split in ["train", "val", "test"]:
        # split_dir = os.path.join(dataset_path, rgb_folder, split)
        split_dir = os.path.join(dataset_path, split, rgb_folder)
        # file_paths = glob(os.path.join(split_dir, "*/*.png"))
        file_paths = glob(os.path.join(split_dir, "*.png"))
        file_paths = sorted(file_paths)
        
        # slice_idx = 3
        # even_slice = len(file_paths) // 4
        # file_paths = file_paths[even_slice * slice_idx : even_slice * (slice_idx + 1)]
        
        # print("Script working on slice", slice_idx)
        for file_path in tqdm(file_paths, desc=split):
            image = cv2.imread(file_path, cv2.IMREAD_COLOR)
            
            H, W = image.shape[:2]
            min_res = min(H,W)
            image_processed = processor(image, 
                                        detect_resolution=min_res, 
                                        image_resolution=min_res, 
                                        output_type="np",
                                        safe=True
                                        )
            image_processed = cv2.resize(image_processed, (W,H), interpolation=cv2.INTER_LINEAR)
            
            dirname = os.path.dirname(file_path)
            save_dir = dirname.replace(rgb_folder, processor_folder)
            os.makedirs(save_dir, exist_ok=True)
            
            filename = os.path.basename(file_path)
            save_path = os.path.join(save_dir, filename)
            cv2.imwrite(save_path, image_processed)
    