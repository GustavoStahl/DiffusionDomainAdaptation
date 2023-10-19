import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR

from torchvision.models import ResNet101_Weights
from torchvision.models.segmentation import deeplabv3_resnet101

# Typing
from torch.optim import Optimizer
from torch import nn
from collections.abc import Iterable
from torch.optim.lr_scheduler import _LRScheduler

from mmseg.core.evaluation.metrics import total_intersect_and_union, total_area_to_metrics

import wandb
import os

import gc
import random
import argparse

from tqdm import tqdm

import numpy as np
import cv2

from enum import Enum, auto

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
    b, c = tensor.size()[:2]
    
    tensor_min = torch.min(tensor.view(b,c,-1), dim=2)[0].view(b,c,1,1)
    tensor_max = torch.max(tensor.view(b,c,-1), dim=2)[0].view(b,c,1,1)
    
    tensor_0_1 = (tensor - tensor_min) / (tensor_max - tensor_min)
    
    array_0_1 = tensor_0_1.permute(0,2,3,1).cpu().numpy()
    
    return (array_0_1 * 255).astype("uint8")

def test(model:nn.Module, 
         test_dataloader:Iterable, 
         epoch:int,
         max_batches:int=None):
    
    class_names = test_dataloader.dataset.CLASSES
    void_idx = test_dataloader.dataset.VOID_IDX
    
    max_batches = len(test_dataloader) if max_batches is None else max_batches
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
            if bidx + 1 == max_batches:
                break
            continue            
            
        image_rgb = tensor2image(image)
        for im_color, pred_mask, true_mask in zip(image_rgb, pred_classes, gt):
            class_labels = {i:c for (i,c) in enumerate(class_names)}
            
            pred_mask = pred_mask.astype("uint8")
            true_mask = true_mask.astype("uint8")
            
            h, w = im_color.shape[:2]
            scale_target = 384 / w
            im_color = cv2.resize(im_color, (None, None), fx=scale_target, fy=scale_target, interpolation=cv2.INTER_LINEAR)
            pred_mask = cv2.resize(pred_mask, (None, None), fx=scale_target, fy=scale_target, interpolation=cv2.INTER_NEAREST)
            true_mask = cv2.resize(true_mask, (None, None), fx=scale_target, fy=scale_target, interpolation=cv2.INTER_NEAREST)

            masks = {"prediction": {"mask_data": pred_mask, "class_labels": class_labels}, 
                     "ground_truth": {"mask_data": true_mask, "class_labels": class_labels}}
            
            visual_pred_list.append(wandb.Image(im_color, masks=masks))    
            
        pbar.update(1)
        if bidx + 1 == max_batches:
            break 
        
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
        
    mean_acc, mean_iou = mean_accs.mean(), mean_ious.mean()
    wandb.define_metric("eval/metrics/macc", summary="max")
    wandb.define_metric("eval/metrics/miou", summary="max")
    wandb.log({"eval/metrics/macc": mean_acc}, step=epoch)                
    wandb.log({"eval/metrics/miou": mean_iou}, step=epoch)        
        
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
    
    return mean_acc, mean_iou
            
def train(model:nn.Module, 
          train_dataloader:Iterable, 
          criterion,
          optimizer:Optimizer, 
          epoch:int,
          max_batches:int=None):
        
    max_batches = len(train_dataloader) if max_batches is None else max_batches
    pbar = tqdm(total=max_batches)
    pbar.set_description(f"train epoch {epoch}")
    
    loss_acum = 0.0
    for bidx, data in enumerate(train_dataloader):
        image, gt = data
        
        image = image.to(device)
        gt = gt.squeeze().to(device)
        
        pred = model(image)["out"]    
                               
        loss = criterion(pred, gt.long())
        loss_acum += loss.item()
               
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        pbar.update(1)
        
        # process first N batches
        if bidx + 1 == max_batches:
            break              
        
    wandb.log({"train/loss/cross_entropy": loss_acum / max_batches}, step=epoch)
        
def loop(model:nn.Module,
         train_dataloader:Iterable, 
         test_dataloader:Iterable, 
         criterion,
         optimizer:Optimizer,
         lr_scheduler:_LRScheduler,
         epochs:int,
         max_batches:int=None):
    
    best_score = 0.
    for epoch in range(epochs):
        model.train()
        train(model, train_dataloader, criterion, optimizer, epoch, max_batches)
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
            torch.save({"model": model.state_dict(),
                        "epoch": epoch,
                        "optimizer": optimizer.state_dict()}, 
                       os.path.join(save_dir, "best.pth"))
        clear_cache()

def main(args):
    
    global device
    device = args.device
    
    set_determinism()
    
    assert len(args.dataset_type) == len(args.dataset_path)
    
    mean_and_std = None
    if len(args.dataset_path) >= 2:
        if args.pretrained is None:
            mean_and_std = [(0.485, 0.456, 0.406), 
                            (0.229, 0.224, 0.225)] # Normalization IMAGENET
        else:
            #! Not ideal, needs to be manually set
            #NOTE: Computed from GTA dataset
            mean_and_std = [(0.42935305, 0.42347938, 0.40977437), 
                            (0.25669742, 0.25097305, 0.24708469)]
            #NOTE: Computed from Cityscapes dataset
            # mean_and_std = [(0.28689553, 0.32513301, 0.28389176), 
            #                 (0.18696375, 0.19017339, 0.18720214)]
    
    test_dataset_list = []
    train_dataset_list = []
    for dataset_type, dataset_path in zip(args.dataset_type, args.dataset_path):
        if DatasetType[dataset_type] == DatasetType.GTA5:
            from gta_dataset import get_dataset
        elif DatasetType[dataset_type] == DatasetType.CITYSCAPES:
            from cityscapes_dataset import get_dataset
        
        train_dataset = get_dataset(dataset_path, 
                                    split="train", 
                                    filter_labels=args.filter_labels,
                                    mean_and_std=mean_and_std)
        test_dataset = get_dataset(dataset_path, 
                                   split="val", 
                                   filter_labels=args.filter_labels,
                                   mean_and_std=mean_and_std)
        
        test_dataset_list.append(test_dataset)
        train_dataset_list.append(train_dataset)
    
    test_dataset = ConcatDataset(test_dataset_list)
    train_dataset = ConcatDataset(train_dataset_list)
    
    #NOTE: work around
    test_dataset.CLASSES = test_dataset_list[0].CLASSES
    test_dataset.VOID_IDX = test_dataset_list[0].VOID_IDX
    
    train_dataloader = DataLoader(train_dataset, 
                                  batch_size=args.batch_size, 
                                  shuffle=True,
                                  num_workers=args.nworkers,
                                  drop_last=True)
    test_dataloader = DataLoader(test_dataset, 
                                 batch_size=args.batch_size, 
                                 shuffle=False,
                                 num_workers=args.nworkers,
                                 drop_last=True)   
    
    num_classes = len(train_dataset_list[0].CLASSES)
    if args.pretrained is None:
        model = deeplabv3_resnet101(num_classes=num_classes, 
                                    weights_backbone=ResNet101_Weights.IMAGENET1K_V1)
    else:
        ckpt = torch.load(args.pretrained, map_location=device)
        # load state dict
        if isinstance(ckpt, dict):
            model = deeplabv3_resnet101(num_classes=num_classes)
            model.load_state_dict(ckpt["model"])
        else:
            model = ckpt    
    
    # if torch.cuda.device_count() > 1:
    #     model = nn.DataParallel(model)
    model.to(device)
    
    # optimizer = Adam(model.parameters(), lr=args.lr)
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=10)
    
    class_weights = torch.tensor(train_dataset_list[0].CLASS_WEIGHTS, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    wandb.init(project="Deeplabv3",
               name=args.exp,
               tags=["train"])  
    wandb.run.log_code(".")
    
    loop(model, 
         train_dataloader, 
         test_dataloader, 
         criterion,
         optimizer, 
         lr_scheduler, 
         args.epochs,
         max_batches=args.max_batches)
    
class DatasetType(Enum):
    GTA5 = auto()
    CITYSCAPES = auto()
    
    @classmethod
    def options(self):
        return [var.name for var in list(self)]    

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp", 
                        default=None, required=False, 
                        help="Experiment name.")
    parser.add_argument("--dataset_path", 
                        required=True,
                        type=str,
                        nargs='*',
                        help="Paths to the datasets.")
    parser.add_argument("--dataset_type", 
                        type=str,
                        nargs='*',
                        choices=DatasetType.options(), 
                        help="Datasets to train with.")      
    parser.add_argument("--pretrained", 
                        type=str,
                        default=None,
                        help="Path to pretrained weights.")     
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
    parser.add_argument("--filter_labels", 
                        action="store_true",
                        help="Whether to remove labels with small area.")           
    return parser.parse_args()    

if __name__ == "__main__":
    args = parse_args()
    main(args)