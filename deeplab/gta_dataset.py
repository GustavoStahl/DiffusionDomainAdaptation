from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import (ToTensor, 
                                    PILToTensor, 
                                    Normalize, 
                                    Compose, 
                                    )

import cv2
import albumentations as A

import numpy as np

import os
from PIL import Image

from typing import Tuple

class GTA5C19Dataset(Dataset):
    
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
    
    CLASS_PIXEL_FREQUENCY = [1421114645, 415143675, 658108900,   
                             83624868,   39562546,  54052758,
                             5864839,    4166278,   336829503,  
                             101127612,  507898488, 25329284,
                             2486187,    172932749, 63343976,   
                             27231208,   2548698,   2241774,
                             308105,     333883107]
    
    # class_weights = 1/log(1.02 + (class_frequencies[:-1] / sum(class_frequencies[:-1])))
    CLASS_WEIGHTS = [ 3.09,  8.44,  5.81, 
                     24.70, 33.74, 30.10, 
                     47.02, 47.98,  9.94, 
                     22.34,  7.18, 38.30, 
                     48.96, 16.10, 28.16, 
                     37.62, 48.93, 49.11, 
                     50.30,   0.0]
        
    def __init__(self, 
                 root_dir, 
                 split="train", 
                 transform=None, 
                 target_transform=None, 
                 pair_transform=None,
                 filter_labels=False,
                 first_n_samples=None):
        assert split in ("train", "val", "test"), "invalid split informed."
        
        #NOTE: map extra indexes to 'void'
        map_func = lambda key: self.CITYSCAPES19IDXS.get(key, self.VOID_IDX)
        
        max_classes = 35 # max amount of classes in GTA 5 dataset
        LUT_cityscapes19 = np.arange(max_classes)
        LUT_cityscapes19 = np.vectorize(map_func, otypes=["uint8"])(LUT_cityscapes19).squeeze()
        
        self.filter_labels = filter_labels                
        self.LUT_cityscapes19 = LUT_cityscapes19
        
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, "images", split)
        self.label_dir = os.path.join(root_dir, "labels", split)
        self.image_filenames = sorted(os.listdir(self.image_dir), 
                                      key=lambda n: int(os.path.splitext(os.path.basename(n))[0]))
        
        if first_n_samples is not None:
            self.image_filenames = self.image_filenames[:first_n_samples]
                
        self.transform = transform
        self.pair_transform = pair_transform
        self.target_transform = target_transform
        
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
        
    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.image_filenames[idx])
        label_name = os.path.join(self.label_dir, self.image_filenames[idx])
        image = Image.open(img_name).convert("RGB")
        label = Image.open(label_name)             
            
        if self.pair_transform is not None:
            pair_transformed = self.pair_transform(image=np.array(image), 
                                                   mask=np.array(label))
            image = Image.fromarray(pair_transformed["image"])
            label = Image.fromarray(pair_transformed["mask"])
            
        label = Image.fromarray(self.convert_labels_format(np.array(label)))
        if self.filter_labels:
            label = Image.fromarray(self.filter_unuseful_labels(np.array(label)))                
                    
        if self.target_transform is not None:
            label = self.target_transform(label)

        if self.transform is not None:
            image = self.transform(image)
            
        return image, label
                
def get_dataloader(dataset_path:str, 
                   split:str="train", 
                   batch_size:int=1, 
                   nworkers:int=0, 
                   filter_labels:bool=False,
                   mean_and_std:Tuple[Tuple[float,float,float], Tuple[float,float,float]]=None,
                   first_n_samples:int=None): 
      
    if mean_and_std is None:
        mean = (0.42935305, 0.42347938, 0.40977437) #NOTE: computed from the training dataset
        std  = (0.25669742, 0.25097305, 0.24708469)
    else:
        mean, std = mean_and_std
       
    pair_transform = None
    if split == "train":
        target_size = (769, 769) 
        standard_size = (1046, 1914) # make sure labels and images have the same size
        pair_transform = A.Compose([A.Resize(*standard_size, 
                                             interpolation=cv2.INTER_LINEAR, 
                                             p=1.0),
                                    A.RandomScale((-0.2, 0.5), 
                                                  interpolation=cv2.INTER_LINEAR, 
                                                  p=0.5),
                                    A.RandomCrop(*target_size, 
                                                 p=1.0),
                                    A.HorizontalFlip(p=0.5)], 
                                   is_check_shapes=False) 
    else:
        standard_size = (1046, 1914)
        pair_transform = A.Compose([A.Resize(*standard_size, 
                                             interpolation=cv2.INTER_LINEAR,
                                             p=1.0)],
                                   is_check_shapes=False)
        
    transform = Compose([ToTensor(), Normalize(mean, std)])
    target_transform = Compose([PILToTensor()])        
    
    dataset = GTA5C19Dataset(dataset_path, 
                             split, 
                             transform, 
                             target_transform, 
                             pair_transform, 
                             filter_labels,
                             first_n_samples)
    dataloader = DataLoader(dataset, 
                            batch_size=batch_size, 
                            shuffle=True if split == "train" else False,
                            num_workers=nworkers,
                            drop_last=True)
    
    return dataloader

def __class_frequency(dataloader):
    label_frequency = np.zeros(len(dataloader.dataset.CLASSES), dtype=np.uint)
    for data in dataloader:
        image, label = data
        unique, counts = np.unique(label, return_counts=True)
        label_frequency[unique] += counts.astype(np.uint)
    return label_frequency

if __name__ == "__main__":
    import torchshow as ts
    dataloader = get_dataloader("/ibex/user/stahlg/datasets/gta/prepared_dataset", 
                                split="val", 
                                nworkers=8,
                                batch_size=4,
                                filter_labels=True)
    from tqdm import tqdm
    for bidx, data in enumerate(tqdm(dataloader)):
        image, label = data
        if 16 not in label:
            continue
        os.makedirs("../results/gta_trains/", exist_ok=True)
        ts.save(image, f"../results/gta_trains/train_image_{bidx}.png")
        # ts.save(label, "../results/gta_dataset_label.png")
        # if bidx < 179: continue
        # if 18 not in label: continue
        # Image.fromarray(np.take(GTA5C19Dataset.PALETTE, label[0].squeeze().numpy(), axis=0).astype("uint8")).save(f"../results/gta_dataset_bicycle_{bicycle_count}.png")
        # bicycle_count += 1
        # if bicycle_count >= 8:
        # break
        # import pdb; pdb.set_trace()