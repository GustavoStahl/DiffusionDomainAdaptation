from torchvision.datasets import Cityscapes
from torch.utils.data import DataLoader

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
    DEPTH = auto()
    LINEART = auto()
    CANNY_AND_SAM = auto()
    
    @classmethod
    def options(self):
        return [var.name for var in list(self)]

class CityscapesC19Dataset(Cityscapes):
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
    
    CLASS_PIXEL_FREQUENCY = [678075191, 87281060,  313104171,  
                             10040422,  13453001,  21293041,   
                             3168423,   9083284,   242893520,  
                             17065188,  48245276,  26570178,   
                             3190590,   131114292, 4619678,   
                             5812564,   5272930,   1884932,   
                             8325041,   110474002]
    
    # class_weights = 1/log(1.02 + (class_frequencies[:-1] / sum(class_frequencies[:-1])))
    CLASS_WEIGHTS = [ 2.76, 14.09,  5.20, 
                     38.73, 35.89, 30.75, 
                     46.07, 39.60,  6.41, 
                     33.32, 20.66, 28.05, 
                     46.04, 10.45, 44.29, 
                     42.93, 43.54, 47.77, 
                     40.33,   0.0]
    
    # CLASS_WEIGHTS = [1., 1., 1.,
    #                  1., 1., 1.,
    #                  1., 1., 1.,
    #                  1., 1., 1.,
    #                  1., 1., 1.,
    #                  1., 1., 1.,
    #                  1., 0.]
    
    def __init__(self, 
                 root,
                 split="train",
                 transform=None,
                 target_transform=None,
                 transforms=None,
                 pair_transform=None,
                 filter_labels=False,
                 first_n_samples=None,
                 condition_type:ConditionType=ConditionType.NONE):
        super().__init__(root, split, "fine", "semantic", transform, target_transform, transforms)
        
        #NOTE: map extra indexes to 'void'
        map_func = lambda key: self.CITYSCAPES19IDXS.get(key, self.VOID_IDX)
        
        max_classes = 35 # max amount of classes in Cityscapes dataset
        LUT_cityscapes19 = np.arange(max_classes)
        LUT_cityscapes19 = np.vectorize(map_func, otypes=["uint8"])(LUT_cityscapes19).squeeze()
        
        if first_n_samples is not None:
            self.images = self.images[:first_n_samples]
            self.targets = self.targets[:first_n_samples]
            
        if condition_type != ConditionType.NONE:
            self.condition_files = self.get_condition_files(condition_type)
        
        self.filter_labels = filter_labels                
        self.pair_transform = pair_transform
        self.LUT_cityscapes19 = LUT_cityscapes19
        self.condition_type = condition_type
        
    def get_condition_files(self, condition_type):
        condition_files = []
        if condition_type in [ConditionType.SAM, 
                              ConditionType.CANNY_AND_SAM, 
                              ConditionType.DEPTH, 
                              ConditionType.LINEART]:
            rgb_folder = "leftImg8bit"
            if condition_type in [ConditionType.SAM, ConditionType.CANNY_AND_SAM]:
                condition_folder = "leftImg8bit_SAM"
            elif condition_type == ConditionType.DEPTH:
                condition_folder = "leftImg8bit_depth"
            elif condition_type == ConditionType.LINEART:
                condition_folder = "leftImg8bit_lineart"
            
            for file_path in self.images:
                dirname = os.path.dirname(file_path)
                filename = os.path.basename(file_path)
                condition_dirname = dirname.replace(rgb_folder, condition_folder)
                condition_files.append(os.path.join(condition_dirname, filename))
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
        image = Image.open(self.images[index]).convert("RGB")
        label = Image.open(self.targets[index][0])
        
        condition = None
        if self.condition_type == ConditionType.SAM:
            condition = Image.open(self.condition_files[index])
        elif self.condition_type in [ConditionType.DEPTH, ConditionType.LINEART]:
            condition = Image.open(self.condition_files[index]).convert("L")
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
        
        # data = {}
        # data["image"] = image
        # data["label"] = label  
        # data["filepath"] = self.images[index]
        # if self.condition_type != ConditionType.NONE:
        #     data["condition"] = condition  
            
        # return data
        
def get_dataset(dataset_path:str, 
                split:str="train", 
                filter_labels:bool=False,
                mean_and_std:Tuple[Tuple[float,float,float], Tuple[float,float,float]]=None,
                first_n_samples:int=None,
                condition_type:ConditionType=ConditionType.NONE):  
         
    if mean_and_std is None:
        mean = (0.28689553, 0.32513301, 0.28389176) #NOTE: computed from training dataset
        std  = (0.18696375, 0.19017339, 0.18720214)
    else:
        mean, std = mean_and_std
       
    pair_transform = None
    if split == "train":
        target_size = (769, 769) 
        pair_transform = A.Compose([A.RandomScale((-0.2, 0.5), 
                                                  interpolation=cv2.INTER_LINEAR, 
                                                  p=0.5),
                                    A.RandomCrop(*target_size, 
                                                 p=1.0),
                                    A.HorizontalFlip(p=0.5)])
    else:
        standard_size = (1046, 1914)
        pair_transform = A.Compose([A.Resize(*standard_size, 
                                             interpolation=cv2.INTER_LINEAR,
                                             p=1.0)],
                                   is_check_shapes=False)        
        
    transform = Compose([ToTensor(), 
                         Normalize(mean, std)])
    target_transform = Compose([PILToTensor()])        
    
    dataset = CityscapesC19Dataset(dataset_path, 
                                   split=split,
                                   transform=transform, 
                                   target_transform=target_transform,
                                   pair_transform=pair_transform,
                                   filter_labels=filter_labels,
                                   first_n_samples=first_n_samples,
                                   condition_type=condition_type)

    return dataset 

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
    dataloader = get_dataloader("/ibex/user/stahlg/datasets/cityscapes", 
                                split="train", 
                                batch_size=32,
                                nworkers=8,
                                condition_type=ConditionType.NONE)
    from tqdm import tqdm
    for data in tqdm(dataloader):
        import pdb; pdb.set_trace()
        print(len(data))
        for i, d in enumerate(data):
            if i == 2:
                Image.fromarray(d.numpy().squeeze()).save(f"../results/cityscapes_with_CANNY_AND_SAM_{i}.png")
            else:
                ts.save(d, f"../results/cityscapes_with_CANNY_AND_SAM_{i}.png")
        break

        # for idx, tensor in enumerate(data):
        #     print(tensor.size())
        #     if idx == 1:
        #         Image.fromarray(np.take(CityscapesC19Dataset.PALETTE, tensor[0].squeeze().numpy(), axis=0).astype("uint8")).save(f"../results/cityscapes_dataset_{idx}.png")
        #         continue            
        #     ts.save(tensor, f"../results/cityscapes_dataset_{idx}.png")
        # break