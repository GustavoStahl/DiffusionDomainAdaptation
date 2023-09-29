import torch
import torch.nn.functional as F
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR

from torchvision.models import ResNet101_Weights
from torchvision.models.segmentation import deeplabv3_resnet101

from gta_dataset import get_dataloader

# Typing
from torch.optim import Optimizer
from torch import nn
from collections.abc import Iterable
from torch.optim.lr_scheduler import _LRScheduler

from mmseg.core import eval_metrics

import wandb
import os

import gc
import random
import argparse

from tqdm import tqdm

import numpy as np

# torchshow for debugging
from torchshow.visualization import auto_unnormalize_image
# necessary for torchshow
import matplotlib; matplotlib.use("Agg")


device = None

def set_determinism():
    # set seed, to be deterministic
    seed = 123
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
def clear_cache():
    gc.collect()
    torch.cuda.empty_cache()  
    
def tensor2image(_tensor):
    tensor = _tensor.detach()
    
    tensor_min = tensor.min(1, keepdim=True)[0]
    tensor_max = tensor.max(1, keepdim=True)[0]
    
    tensor_0_1 = (tensor - tensor_min / (tensor_max - tensor_min))
    
    array_0_1 = tensor_0_1.permute(0,2,3,1).cpu().numpy()
    
    return (array_0_1 * 255).astype("uint8")

def test(model:nn.Module, 
         test_dataloader:Iterable, 
         epoch:int,
         max_batches:int=None):
    
    class_names = test_dataloader.dataset.CLASSES
    
    mean_iou = np.zeros(len(class_names), dtype=np.float32)
    mean_acc = np.zeros(len(class_names), dtype=np.float32)
    
    max_batches = len(test_dataloader) if max_batches is None else max_batches
    pbar = tqdm(total=max_batches)
    pbar.set_description(f"test epoch {epoch}")
    
    visual_pred_list = []
    for bidx, data in enumerate(test_dataloader):
        clear_cache()
        
        image, gt = data
        
        image = image.to(device)
        
        with torch.no_grad():
            pred = model(image)["out"]
            
        pred_classes = pred.argmax(1).cpu().numpy()
        gt = gt.squeeze().numpy()
        
        ret_metrics = eval_metrics(pred_classes,
                                   gt,
                                   len(class_names),
                                   ignore_index=len(class_names)-1,
                                   metrics=["mIoU"],
                                   label_map=None)    
        
        mean_acc += np.nan_to_num(ret_metrics["Acc"])
        mean_iou += np.nan_to_num(ret_metrics["IoU"])        
            
        # save results every N batches
        if bidx % (max(max_batches // 4, 1)) != 0: 
            clear_cache()
            pbar.update(1)
            if bidx + 1 == max_batches:
                break
            continue            
            
        image_rgb = tensor2image(image)
        for im_color, pred_mask, true_mask in zip(image_rgb, pred_classes, gt):
            class_labels = {i:c for (i,c) in enumerate(class_names)}
            
            pred_mask = pred_mask.astype("uint8")
            true_mask = true_mask.astype("uint8")

            masks = {"prediction": {"mask_data": pred_mask, "class_labels": class_labels}, 
                     "ground_truth": {"mask_data": true_mask, "class_labels": class_labels}}
            
            visual_pred_list.append(wandb.Image(im_color, masks=masks))    
            
        clear_cache()
        pbar.update(1)
        if bidx + 1 == max_batches:
            break   
        
    mean_acc /= max_batches
    mean_iou /= max_batches
    for class_name, acc, iou in zip(class_names, mean_acc, mean_iou):
        wandb.log({f"eval/metrics/acc/{class_name}": acc}, step=epoch)                
        wandb.log({f"eval/metrics/iou/{class_name}": iou}, step=epoch)
        
    for metric_name, metric_values in zip(["Acc", "IoU"], [mean_acc, mean_iou]):
        # Create a wandb.Table with columns for class names and metric values
        table_data = list(zip(class_names, metric_values))
        table = wandb.Table(data=table_data, columns=["class", metric_name])

        # Create a bar graph using wandb.plot.bar
        plot = wandb.plot.bar(table, label="class", value=metric_name, title=f'Per Class {metric_name}')

        # Log the table and plot with a specific topic name
        topic_name = f"eval/metrics/{metric_name.lower()}"
        wandb.log({f"{topic_name}_plot": plot}, step=epoch)           
        
    wandb.log({"eval/images": visual_pred_list}, step=epoch)
    
    return mean_acc.mean(), mean_iou.mean()
            
def train(model:nn.Module, 
          train_dataloader:Iterable, 
          optimizer:Optimizer, 
          epoch:int,
          max_batches:int=None):
        
    class_names = train_dataloader.dataset.CLASSES
    
    mean_iou = np.zeros(len(class_names), dtype=np.float32)
    mean_acc = np.zeros(len(class_names), dtype=np.float32)
    
    max_batches = len(train_dataloader) if max_batches is None else max_batches
    pbar = tqdm(total=max_batches)
    pbar.set_description(f"train epoch {epoch}")
    
    for bidx, data in enumerate(train_dataloader):
        clear_cache()
        
        image, gt = data
        
        gt = gt.squeeze().to(device)
        image = image.to(device)
        
        pred = model(image)["out"]    
                               
        loss = F.cross_entropy(pred, gt.long())
        wandb.log({"train/loss/cross_entropy": loss.item()}, step=epoch)
        
        pred_classes = pred.detach().argmax(1).cpu().numpy()
        ret_metrics = eval_metrics(pred_classes,
                                   gt.cpu().numpy(),
                                   len(class_names),
                                   ignore_index=len(class_names)-1,
                                   metrics=["mIoU"],
                                   label_map=None)    
        
        mean_acc += np.nan_to_num(ret_metrics["Acc"])
        mean_iou += np.nan_to_num(ret_metrics["IoU"])
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        clear_cache()
        pbar.update(1)
        
        # process first N batches
        if bidx + 1 == max_batches:
            break   
        
    mean_acc /= max_batches
    mean_iou /= max_batches
    for class_name, acc, iou in zip(class_names, mean_acc, mean_iou):
        wandb.log({f"train/metrics/acc/{class_name}": acc}, step=epoch)                
        wandb.log({f"train/metrics/iou/{class_name}": iou}, step=epoch)
        
    for metric_name, metric_values in zip(["Acc", "IoU"], [mean_acc, mean_iou]):
        # Create a wandb.Table with columns for class names and metric values
        table_data = list(zip(class_names, metric_values))
        table = wandb.Table(data=table_data, columns=["class", metric_name])

        # Create a bar graph using wandb.plot.bar
        plot = wandb.plot.bar(table, label="class", value=metric_name, title=f'Per Class {metric_name}')

        # Log the table and plot with a specific topic name
        topic_name = f"train/metrics/{metric_name.lower()}"
        wandb.log({f"{topic_name}_plot": plot}, step=epoch)               
        
def loop(model:nn.Module,
         train_dataloader:Iterable, 
         test_dataloader:Iterable, 
         optimizer:Optimizer,
         lr_scheduler:_LRScheduler,
         epochs:int,
         max_batches:int=None):
    
    best_score = 0.
    for epoch in range(epochs):
        model.train()
        train(model, train_dataloader, optimizer, epoch, max_batches)
        if epoch % 5 != 0:
            continue
        model.eval()
        acc, iou = test(model, test_dataloader, epoch, max_batches)
        eval_score = (acc + iou) / 2.
        lr_scheduler.step()
        if eval_score > best_score:
            print("[INFO] Better weights found! Saving them...")
            save_dir = os.path.join(wandb.run.dir, "weights")
            os.makedirs(save_dir, exist_ok=True)
            torch.save(model, os.path.join(save_dir, "best.pth"))

def main(args):
    
    global device
    device = args.device
    
    set_determinism()
    
    train_dataloader = get_dataloader(args.dataset_path, 
                                      split="train", 
                                      batch_size=args.batch_size,
                                      nworkers=args.nworkers)
    test_dataloader = get_dataloader(args.dataset_path, 
                                     split="val", 
                                     batch_size=args.batch_size, 
                                     nworkers=args.nworkers)
    
    num_classes = len(train_dataloader.dataset.CLASSES)
    model = deeplabv3_resnet101(num_classes=num_classes, 
                                weights_backbone=ResNet101_Weights.IMAGENET1K_V1)
    model.to(device)
    
    optimizer = Adam(model.parameters(), lr=args.lr)
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=10)
    
    wandb.init(project="Deeplabv3",
               name=args.exp)   
    
    loop(model, 
         train_dataloader, 
         test_dataloader, 
         optimizer, 
         lr_scheduler, 
         args.epochs,
         max_batches=args.max_batches)
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp", 
                        default=None, required=False, 
                        help="Experiment name.")
    parser.add_argument("--dataset_path", 
                        required=True,
                        type=str,
                        help="Path to the dataset.")
    parser.add_argument("--lr", 
                        default=1e-4,
                        type=float,
                        help="Training learning rate.")    
    parser.add_argument("--epochs", 
                        type=int,
                        default=100, 
                        help="Training epochs.")   
    parser.add_argument("--batch_size", 
                        type=int,
                        default=8, 
                        help="Number of samples in batch.")         
    parser.add_argument("--max_batches", 
                        type=int,
                        default=None, 
                        help="Number of samples in batch.")     
    parser.add_argument("--nworkers", 
                        type=int,
                        default=0, 
                        help="Number of workers for the dataloader.")              
    parser.add_argument("--device", 
                        type=str,
                        default="cuda:0", 
                        help="Which device to use for the training.")       
    return parser.parse_args()    

if __name__ == "__main__":
    args = parse_args()
    main(args)