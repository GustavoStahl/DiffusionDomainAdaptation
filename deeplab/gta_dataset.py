import torch
from torch.utils.data import DataLoader, Dataset
import datasets
from torchvision.transforms import ToTensor, PILToTensor, Normalize, Compose, Resize, InterpolationMode

import numpy as np

import os
from PIL import Image

class GTA5C19Dataset(Dataset):
    
    # Classes: 19
    CLASSES = ["road",    "sidewalk", "building", 
               "wall",    "fence",    "pole", 
               "light",   "sign",     "vegetation", 
               "terrain", "sky",      "person", 
               "rider",   "car",      "truck", 
               "bus",     "train",    "motocycle", 
               "bicycle", "unknown"]
    
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
        
    def __init__(self, root_dir, split="train", transform=None, target_transform=None):
        assert split in ("train", "val", "test"), "invalid split informed."
        
        #NOTE: map extra indexes to 'unkown'
        unknown_idx = list(self.CITYSCAPES19IDXS.values())[-1] + 1
        map_func = lambda key: self.CITYSCAPES19IDXS.get(key, unknown_idx)
        
        max_classes = 35 # max amount of classes in GTA 5 dataset
        LUT_cityscapes19 = np.arange(max_classes)
        LUT_cityscapes19 = np.vectorize(map_func, otypes=["uint8"])(LUT_cityscapes19).squeeze()
        
        self.LUT_cityscapes19 = LUT_cityscapes19
        
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, "images", split)
        self.label_dir = os.path.join(root_dir, "labels", split)
        self.image_filenames = sorted(os.listdir(self.image_dir), 
                                      key=lambda n: int(os.path.splitext(os.path.basename(n))[0]))
                
        self.transform = transform
        self.target_transform = target_transform
        
    def convert_labels_format(self, label):
        label_mapped = np.take(self.LUT_cityscapes19, label)
        return Image.fromarray(label_mapped)
        
    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.image_filenames[idx])
        label_name = os.path.join(self.label_dir, self.image_filenames[idx])
        image = Image.open(img_name).convert("RGB")
        label = Image.open(label_name)
        
        label = self.convert_labels_format(np.array(label))
        
        if self.target_transform is not None:
            label = self.target_transform(label)

        if self.transform is not None:
            image = self.transform(image)
            
        return image, label
                
def get_dataloader(dataset_path, split="train", batch_size=1, nworkers=0):   
    target_size = (769, 769) 
    mean = (0.42935305, 0.42347938, 0.40977437)
    std  = (0.25669742, 0.25097305, 0.24708469)
    
    transform = Compose([ToTensor(), Resize(target_size, InterpolationMode.BILINEAR), Normalize(mean, std)])
    target_transform = Compose([PILToTensor(), Resize(target_size, InterpolationMode.NEAREST)])
    
    dataset = GTA5C19Dataset(dataset_path, split, transform, target_transform)
    dataloader = DataLoader(dataset, 
                            batch_size=batch_size, 
                            shuffle=True if split == "train" else False,
                            num_workers=nworkers,
                            drop_last=True)
    
    return dataloader

if __name__ == "__main__":
    import torchshow as ts
    dataloader = get_dataloader("/home/stahlg/gta_dataset/prepared_dataset", 
                                split="val", 
                                batch_size=8)
    for data in dataloader:
        for idx, tensor in enumerate(data):
            print(tensor.size())
            ts.save(tensor, f"../results/gta_dataset_{idx}.png")
        break