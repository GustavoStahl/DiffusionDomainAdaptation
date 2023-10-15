# get masks
# join them into one single image
# assign a random palette for each mask index
# Image.fromarray(image_with_palette).convert("P", palette=Image.ADAPTIVE, colors=256)
# save to disk

from glob import glob
import os

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import cv2
from PIL import Image
import numpy as np
from tqdm import tqdm

if __name__ == "__main__":
    device = "cuda"
    # dataset_path = "/ibex/user/stahlg/datasets/cityscapes"
    dataset_path = "/ibex/user/stahlg/datasets/kitti_semantic"
    
    # rgb_folder = "leftImg8bit"
    rgb_folder = "image_2"
    # segmentation_folder = "leftImg8bit_SAM"
    segmentation_folder = "sam_masks"
    
    sam = sam_model_registry["default"](checkpoint="segment-anything/sam_vit_h_4b8939.pth").to(device)
    mask_generator = SamAutomaticMaskGenerator(sam, stability_score_thresh=0.75)

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
            image = cv2.imread(file_path)
            image = image[...,::-1] # BGR ➙ RGB
            
            masks = mask_generator.generate(image)
            
            mask_shape = masks[0]["segmentation"].shape
            canvas = np.zeros(mask_shape[:2] + (3,), dtype=np.uint8)
            masks = sorted(masks, key=lambda m: m["area"], reverse=True)
            
            for mask in masks:
                mask = mask["segmentation"].astype(bool)
                canvas[mask] = np.random.randint(0, 255, (3,), dtype=np.uint8)
                
            image_palette = Image.fromarray(canvas)
            image_palette = image_palette.convert("P", palette=Image.ADAPTIVE, colors=256)
            
            dirname = os.path.dirname(file_path)
            save_dir = dirname.replace(rgb_folder, segmentation_folder)
            os.makedirs(save_dir, exist_ok=True)
            
            filename = os.path.basename(file_path)
            save_path = os.path.join(save_dir, filename)
            image_palette.save(save_path)
    