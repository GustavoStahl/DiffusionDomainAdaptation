from torch.utils.data import DataLoader, Dataset

from torchvision.transforms import (ToTensor, 
                                    PILToTensor, 
                                    Normalize, 
                                    Compose, 
                                    )

import cv2
import albumentations as A

import os

import numpy as np
from PIL import Image

from typing import Tuple
from enum import Enum, auto

class ConditionType(Enum):
    NONE = auto()
    CANNY = auto()
    SAM = auto()
    CANNY_AND_SAM = auto()
    
    @classmethod
    def options(self):
        return [var.name for var in list(self)]

class KittiC19Dataset(Dataset):
    # Classes: 19
    CLASSES = ["road",    "sidewalk", "building", 
               "wall",    "fence",    "pole", 
               "light",   "sign",     "vegetation", 
               "terrain", "sky",      "person", 
               "rider",   "car",      "truck", 
               "bus",     "train",    "motocycle", 
               "bicycle", "void"]
    
    PALETTE = [(128,  64, 128), (244,  35, 232), ( 70,  70,  70), 
               (102, 102, 156), (190, 153, 153), (153, 153, 153), 
               (250, 170,  30), (220, 220,   0), (107, 142,  35), 
               (152, 251, 152), ( 70, 130, 180), (220,  20,  60), 
               (255,   0,   0), (  0,   0, 142), (  0,   0,  70), 
               (  0,  60, 100), (  0,  80, 100), (  0,   0, 230), 
               (119,  11,  32), (  0,   0,   0)]
    
    CITYSCAPES19IDXS = {
         7:  0,
         8:  1,
        11:  2,
        12:  3,
        13:  4,
        17:  5,
        19:  6,
        20:  7,
        21:  8,
        22:  9,
        23: 10,
        24: 11,
        25: 12,
        26: 13,
        27: 14,
        28: 15,
        31: 16,
        32: 17,
        33: 18,
    }
    VOID_IDX = 19
        
    CLASS_WEIGHTS = [1., 1., 1.,
                     1., 1., 1.,
                     1., 1., 1.,
                     1., 1., 1.,
                     1., 1., 1.,
                     1., 1., 1.,
                     1., 0.]
    
    def __init__(self, 
                 root_dir,
                 split="training",
                 transform=None,
                 target_transform=None,
                 pair_transform=None,
                 filter_labels=False,
                 first_n_samples=None,
                 condition_type:ConditionType=ConditionType.NONE):
        assert split == "training", "invalid split informed."
        
        #NOTE: map extra indexes to 'void'
        map_func = lambda key: self.CITYSCAPES19IDXS.get(key, self.VOID_IDX)
        
        max_classes = 35 # max amount of classes in Cityscapes dataset
        LUT_cityscapes19 = np.arange(max_classes)
        LUT_cityscapes19 = np.vectorize(map_func, otypes=["uint8"])(LUT_cityscapes19).squeeze()
        
        self.filter_labels = filter_labels                
        self.LUT_cityscapes19 = LUT_cityscapes19
        
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, split, "image_2")
        self.label_dir = os.path.join(root_dir, split, "semantic")
        self.image_filenames = sorted(os.listdir(self.image_dir), 
                                      key=lambda n: int(os.path.splitext(os.path.basename(n))[0]))        
        
        if first_n_samples is not None:
            self.image_filenames = self.image_filenames[:first_n_samples]
            
        if condition_type != ConditionType.NONE:
            self.condition_files = self.get_condition_files(condition_type)
        
        self.transform = transform
        self.pair_transform = pair_transform
        self.target_transform = target_transform

        self.condition_type = condition_type
        
    def get_condition_files(self, condition_type):
        condition_files = []
        if condition_type in [ConditionType.SAM, ConditionType.CANNY_AND_SAM]:
            rgb_folder = "image_2"
            sam_folder = "sam_masks"
            
            for filename in self.image_filenames:
                sam_dirname = self.image_dir.replace(rgb_folder, sam_folder)
                condition_files.append(os.path.join(sam_dirname, filename))
        return condition_files
        
    def convert_labels_format(self, label):
        return np.take(self.LUT_cityscapes19, label)
    
    def filter_unuseful_labels(self, label):
        indexes, counts = np.unique(label, return_counts=True)
        unuseful_indexes = indexes[counts < 150]
        if not unuseful_indexes.any():
            return label
        unuseful_pixels = np.isin(label, unuseful_indexes)
        label[unuseful_pixels] = self.VOID_IDX
        return label 
    
    def load_files(self, index):
        img_name = os.path.join(self.image_dir, self.image_filenames[index])
        label_name = os.path.join(self.label_dir, self.image_filenames[index])
        image = Image.open(img_name).convert("RGB")
        label = Image.open(label_name)  
        
        condition = None
        if self.condition_type == ConditionType.SAM:
            condition = Image.open(self.condition_files[index])
        elif self.condition_type == ConditionType.CANNY:
            image_bgr = np.array(image)[...,::-1]
            condition = Image.fromarray(cv2.Canny(image_bgr, 50, 100))   
        elif self.condition_type == ConditionType.CANNY_AND_SAM:
            image_bgr = np.array(image)[...,::-1]
            canny_condition = cv2.Canny(image_bgr, 50, 100)   
            sam_condition = np.array(Image.open(self.condition_files[index]).convert("L"))
            
            sam_canny = cv2.Canny(sam_condition, 50, 100)   
                    
            contours, _ = cv2.findContours(sam_condition, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            sam_contours = np.zeros_like(canny_condition)
            cv2.drawContours(sam_contours, contours, -1, 255, thickness=1)   
            
            condition = (sam_contours) | (sam_canny) | (canny_condition)
            condition = Image.fromarray(condition)
            
        return image, label, condition     
    
    def __len__(self):
        return len(self.image_filenames)    
    
    def __getitem__(self, index):
        image, label, condition = self.load_files(index)
        
        if self.pair_transform is not None:
            masks = [np.array(label)]
            
            if self.condition_type != ConditionType.NONE:
                masks += [np.array(condition)]
            
            pair_transformed = self.pair_transform(image=np.array(image), 
                                                   masks=masks)
            
            image = Image.fromarray(pair_transformed["image"])
            if self.condition_type == ConditionType.NONE:
                label = Image.fromarray(pair_transformed["masks"][0])
            else:
                label, condition = [Image.fromarray(m) for m in pair_transformed["masks"]]
            
        label = Image.fromarray(self.convert_labels_format(np.array(label)))
        if self.filter_labels:
            label = Image.fromarray(self.filter_unuseful_labels(np.array(label)))            
            
        if self.target_transform is not None:
            label = self.target_transform(label)
            if self.condition_type != ConditionType.NONE:
                condition = self.target_transform(condition)

        if self.transform is not None:
            image = self.transform(image)

        if self.condition_type == ConditionType.NONE:
            return image, label
        else:
            return image, label, condition
        
def get_dataloader(dataset_path:str, 
                   split:str="training", 
                   batch_size:int=1, 
                   nworkers:int=0, 
                   filter_labels:bool=False,
                   mean_and_std:Tuple[Tuple[float,float,float], Tuple[float,float,float]]=None,
                   first_n_samples:int=None,
                   condition_type:ConditionType=ConditionType.NONE):  
         
    if mean_and_std is None:
        mean = (0.5, 0.5, 0.5) #NOTE: computed from training dataset
        std  = (0.5, 0.5, 0.5)
    else:
        mean, std = mean_and_std
       
    standard_size = (384, 1248)
    pair_transform = A.Compose([A.Resize(*standard_size, 
                                         interpolation=cv2.INTER_LINEAR,
                                         p=1.0)],
                                is_check_shapes=False)
    
    transform = Compose([ToTensor(), 
                         Normalize(mean, std)])
    target_transform = Compose([PILToTensor()])        
    
    dataset = KittiC19Dataset(dataset_path, 
                              split=split,
                              transform=transform, 
                              target_transform=target_transform,
                              pair_transform=pair_transform,
                              filter_labels=filter_labels,
                              first_n_samples=first_n_samples,
                              condition_type=condition_type)
    dataloader = DataLoader(dataset, 
                            batch_size=batch_size, 
                            shuffle=False,#True if split == "train" else False,
                            num_workers=nworkers,
                            drop_last=True)   
    return dataloader 

def __class_frequency(dataloader):
    label_frequency = np.zeros(len(dataloader.dataset.CLASSES), dtype=np.uint)
    from tqdm import tqdm
    for data in tqdm(dataloader):
        image, label = data
        unique, counts = np.unique(label, return_counts=True)
        label_frequency[unique] += counts.astype(np.uint)
    return label_frequency

def set_determinism():
    # set seed, to be deterministic
    import random, numpy as np, torch
    seed = 123
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

if __name__ == "__main__":
    import torchshow as ts
    set_determinism()
    dataloader = get_dataloader("/ibex/user/stahlg/datasets/kitti_semantic", 
                                split="training", 
                                batch_size=1,
                                nworkers=1,
                                condition_type=ConditionType.NONE)
    from tqdm import tqdm
    for data in tqdm(dataloader):
        print(len(data))
        for i, d in enumerate(data):
            ts.save(d, f"../results/kitty_{i}.png")
        break

        # for idx, tensor in enumerate(data):
        #     print(tensor.size())
        #     if idx == 1:
        #         Image.fromarray(np.take(CityscapesC19Dataset.PALETTE, tensor[0].squeeze().numpy(), axis=0).astype("uint8")).save(f"../results/cityscapes_dataset_{idx}.png")
        #         continue            
        #     ts.save(tensor, f"../results/cityscapes_dataset_{idx}.png")
        # break