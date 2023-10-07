from torchvision.datasets import Cityscapes
from torch.utils.data import DataLoader

from torchvision.transforms import (ToTensor, 
                                    PILToTensor, 
                                    Normalize, 
                                    Compose, 
                                    )

import cv2
import albumentations as A

import numpy as np
from PIL import Image

from typing import Tuple

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
    
    CLASS_WEIGHTS = [1, 1, 1,
                     1, 1, 1,
                     1, 1, 1,
                     1, 1, 1,
                     1, 1, 1,
                     1, 1, 1,
                     1, 0]
    
    def __init__(self, 
                 root,
                 split="train",
                 transform=None,
                 target_transform=None,
                 transforms=None,
                 pair_transform=None,
                 filter_labels=False):
        super().__init__(root, split, "fine", "semantic", transform, target_transform, transforms)
    
        #NOTE: map extra indexes to 'void'
        map_func = lambda key: self.CITYSCAPES19IDXS.get(key, self.VOID_IDX)
        
        max_classes = 35 # max amount of classes in Cityscapes dataset
        LUT_cityscapes19 = np.arange(max_classes)
        LUT_cityscapes19 = np.vectorize(map_func, otypes=["uint8"])(LUT_cityscapes19).squeeze()
        
        self.filter_labels = filter_labels                
        self.pair_transform = pair_transform
        self.LUT_cityscapes19 = LUT_cityscapes19
        
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
    
    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert("RGB")
        label = Image.open(self.targets[index][0])
        
        label = Image.fromarray(self.convert_labels_format(np.array(label)))
        if self.filter_labels:
            label = Image.fromarray(self.filter_unuseful_labels(np.array(label)))
        
        if self.pair_transform is not None:
            pair_transformed = self.pair_transform(image=np.array(image), 
                                                   mask=np.array(label))
            image = Image.fromarray(pair_transformed["image"])
            label = Image.fromarray(pair_transformed["mask"])
            
        if self.transforms is not None:
            image, label = self.transforms(image, label)

        return image, label

def get_dataloader(dataset_path:str, 
                   split:str="train", 
                   batch_size:int=1, 
                   nworkers:int=0, 
                   filter_labels:bool=False,
                   mean_and_std:Tuple[Tuple[float,float,float], Tuple[float,float,float]]=None):  
         
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
        
    transform = Compose([ToTensor(), 
                         Normalize(mean, std)])
    target_transform = Compose([PILToTensor()])        
    
    dataset = CityscapesC19Dataset(dataset_path, 
                                   split=split, 
                                   transform=transform, 
                                   target_transform=target_transform,
                                   pair_transform=pair_transform,
                                   filter_labels=filter_labels)
    dataloader = DataLoader(dataset, 
                            batch_size=batch_size, 
                            shuffle=True if split == "train" else False,
                            num_workers=nworkers,
                            drop_last=True)   
    return dataloader 

if __name__ == "__main__":
    import torchshow as ts
    dataloader = get_dataloader("/ibex/user/stahlg/datasets/cityscapes", 
                                split="test", 
                                batch_size=8,
                                nworkers=4)
    from tqdm import tqdm
    for data in tqdm(dataloader):
        continue

        # for idx, tensor in enumerate(data):
        #     print(tensor.size())
        #     if idx == 1:
        #         Image.fromarray(np.take(CityscapesC19Dataset.PALETTE, tensor[0].squeeze().numpy(), axis=0).astype("uint8")).save(f"../results/cityscapes_dataset_{idx}.png")
        #         continue            
        #     ts.save(tensor, f"../results/cityscapes_dataset_{idx}.png")
        # break