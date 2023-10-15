import torch

from torchvision.models.segmentation import deeplabv3_resnet101

# Typing
from torch import nn
from collections.abc import Iterable

from mmseg.core.evaluation.metrics import total_intersect_and_union, total_area_to_metrics

import wandb
import random
import argparse

from tqdm import tqdm

import numpy as np
import cv2
import os

from enum import Enum, auto

device = None

def set_determinism():
    # set seed, to be deterministic
    seed = 123
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
      
def tensor2image(_tensor):
    tensor = _tensor.detach()
    b, c = tensor.size()[:2]
    
    tensor_min = torch.min(tensor.view(b,c,-1), dim=2)[0].view(b,c,1,1)
    tensor_max = torch.max(tensor.view(b,c,-1), dim=2)[0].view(b,c,1,1)
    
    tensor_0_1 = (tensor - tensor_min) / (tensor_max - tensor_min)
    
    array_0_1 = tensor_0_1.permute(0,2,3,1).cpu().numpy()
    
    return (array_0_1 * 255).astype("uint8")

def test(model:nn.Module, 
         test_dataloader:Iterable, 
         epoch:int=0):
    
    class_names = test_dataloader.dataset.CLASSES
    void_idx = test_dataloader.dataset.VOID_IDX
        
    max_batches = len(test_dataloader)
    pbar = tqdm(total=max_batches)
    pbar.set_description(f"test epoch {epoch}")
    
    num_classes = len(class_names)
    total_area_intersect = torch.zeros((num_classes, ), dtype=torch.float64)
    total_area_union = torch.zeros((num_classes, ), dtype=torch.float64)
    total_area_pred_label = torch.zeros((num_classes, ), dtype=torch.float64)
    total_area_label = torch.zeros((num_classes, ), dtype=torch.float64)    
    
    visual_pred_list = []
    for bidx, data in enumerate(test_dataloader):
        
        image, gt = data
        
        image = image.to(device)
        
        with torch.no_grad():
            pred = model(image)["out"]
            
        pred_classes = pred.argmax(1).cpu().numpy()
        gt = gt.squeeze().numpy()  
        
        area_intersect, area_union, \
        area_pred_label, area_label = total_intersect_and_union(pred_classes, 
                                                                gt, 
                                                                num_classes, 
                                                                void_idx, 
                                                                label_map=None,
                                                                reduce_zero_label=False)
        total_area_intersect += area_intersect
        total_area_union += area_union
        total_area_pred_label += area_pred_label
        total_area_label += area_label
                           
        # save results every N batches
        if bidx % (max(max_batches // 4, 1)) != 0: 
            pbar.update(1)
            continue            
            
        image_rgb = tensor2image(image)
        for im_color, pred_mask, true_mask in zip(image_rgb, pred_classes, gt):
            class_labels = {i:c for (i,c) in enumerate(class_names)}
            
            pred_mask = pred_mask.astype("uint8")
            true_mask = true_mask.astype("uint8")
            
            height = im_color.shape[0]
            scale_target = 384 / height
            im_color = cv2.resize(im_color, (None, None), fx=scale_target, fy=scale_target, interpolation=cv2.INTER_LINEAR)
            pred_mask = cv2.resize(pred_mask, (None, None), fx=scale_target, fy=scale_target, interpolation=cv2.INTER_NEAREST)
            true_mask = cv2.resize(true_mask, (None, None), fx=scale_target, fy=scale_target, interpolation=cv2.INTER_NEAREST)            

            masks = {"prediction": {"mask_data": pred_mask, "class_labels": class_labels}, 
                     "ground_truth": {"mask_data": true_mask, "class_labels": class_labels}}
            
            visual_pred_list.append(wandb.Image(im_color, masks=masks))    
            
        pbar.update(1)
        
    ret_metrics = total_area_to_metrics(total_area_intersect, 
                                        total_area_union,
                                        total_area_pred_label,
                                        total_area_label, 
                                        metrics=["mIoU"], 
                                        nan_to_num=0,
                                        beta=1)       
           
    mean_accs = ret_metrics["Acc"]
    mean_ious = ret_metrics["IoU"]       
        
    # Remove the void class from the metrics
    mean_accs = np.delete(mean_accs, void_idx)
    mean_ious = np.delete(mean_ious, void_idx)
    class_names = list(np.delete(np.array(class_names, dtype=object), void_idx))
        
    for class_name, acc, iou in zip(class_names, mean_accs, mean_ious):
        wandb.log({f"eval/metrics/acc/{class_name}": acc}, step=epoch)                
        wandb.log({f"eval/metrics/iou/{class_name}": iou}, step=epoch)
        
    for metric_name, metric_values in zip(["Acc", "IoU"], [mean_accs, mean_ious]):
        # Create a wandb.Table with columns for class names and metric values
        table_data = list(zip(class_names, metric_values))
        table = wandb.Table(data=table_data, columns=["class", metric_name])

        # Create a bar graph using wandb.plot.bar
        plot = wandb.plot.bar(table, label="class", value=metric_name, title=f'Per Class {metric_name}')

        # Log the table and plot with a specific topic name
        topic_name = f"eval/metrics/{metric_name.lower()}"
        wandb.log({f"{topic_name}_plot": plot}, step=epoch)           
        
    wandb.log({"eval/images": visual_pred_list}, step=epoch)

def main(args):
    
    global device
    device = args.device
    
    set_determinism()
    
    if DatasetType[args.dataset_type] == DatasetType.GTA5:
        from gta_dataset import get_dataloader
    elif DatasetType[args.dataset_type] == DatasetType.CITYSCAPES:
        from cityscapes_dataset import get_dataloader
    elif DatasetType[args.dataset_type] == DatasetType.KITTI:
        from kitti_dataset import get_dataloader
    
    test_dataloader = get_dataloader(args.dataset_path, 
                                     split=args.dataset_split, 
                                     batch_size=args.batch_size, 
                                     nworkers=args.nworkers,
                                     filter_labels=args.filter_labels)

    ckpt = torch.load(args.ckpt_path, map_location=device)
    # load state dict
    if isinstance(ckpt, dict):
        num_classes = len(test_dataloader.dataset.CLASSES)
        model = deeplabv3_resnet101(num_classes=num_classes)
        model.load_state_dict(ckpt["model"])
        epoch = ckpt.get("epoch", 0)
    else:
        model = ckpt
        epoch = 0
    
    model.to(device)
    model.eval()
    
    wandb.init(project="Deeplabv3",
               name=f"{args.exp}_{args.dataset_type}_{args.dataset_split}",
               tags=["test"],
               dir=os.environ.get("WANDB_DIR"))   
    
    test(model, test_dataloader, epoch=epoch)
    
class DatasetType(Enum):
    GTA5 = auto()
    CITYSCAPES = auto()
    KITTI = auto()
    
    @classmethod
    def options(self):
        return [var.name for var in list(self)]    

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp", 
                        default=None, required=False, 
                        help="Experiment name.")    
    parser.add_argument("--ckpt_path", 
                        required=True,
                        type=str,
                        help="Path to the pretrained weights.")   
    parser.add_argument("--dataset_path", 
                        required=True,
                        type=str,
                        help="Path to the dataset.")
    parser.add_argument("--dataset_split", 
                        default="val",
                        type=str,
                        help="Which split to use.")      
    parser.add_argument("--dataset_type", 
                        type=str,
                        default=DatasetType.GTA5.name,
                        choices=DatasetType.options(), 
                        help="Dataset to eval with.")      
    parser.add_argument("--batch_size", 
                        type=int,
                        default=8, 
                        help="Number of samples in batch.")         
    parser.add_argument("--nworkers", 
                        type=int,
                        default=0, 
                        help="Number of workers for the dataloader.")              
    parser.add_argument("--device", 
                        type=str,
                        default="cpu", 
                        help="Which device to use for the testing.")  
    parser.add_argument("--filter_labels", 
                        action="store_true",
                        help="Whether to remove labels with small area.")           
    return parser.parse_args()    

if __name__ == "__main__":
    args = parse_args()
    main(args)