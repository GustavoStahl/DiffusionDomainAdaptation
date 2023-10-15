# core libraries
import os
import random

# computer vision and arrays
import cv2
import numpy as np

# deep learning
import torch
from mmseg.core.evaluation.metrics import total_intersect_and_union, total_area_to_metrics
from torchvision.transforms import Normalize

# custom scripts
from controlnet_augmentation import ControlNetAugmentation
from t2i_augmentation import T2IAdapterAugmentation
from base_augmentation import SchedulerType, DenoiseType

# visualization
import wandb
import argparse
from tqdm import tqdm

# torchshow for debugging
import torchshow as ts
# necessary for torchshow
import matplotlib; matplotlib.use("Agg")

from dataclasses import dataclass

from torchvision.models.segmentation import deeplabv3_resnet101
from deeplab.cityscapes_dataset import ConditionType    

from enum import Enum, auto

device = None

@dataclass
class Config(object):
    #NOTE hardcoded prompt
    PROMPT = ""
    # input shape
    CNET_INSHAPE = (768,768)
    DEEPLAB_INSHAPE = (769,769)
    # normalization values
    CNET_MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32) # convert [0,1]➙[-1,1]
    CNET_STD  = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    # DEEPLAB_MEAN = np.array([0.28689553, 0.32513301, 0.28389176], dtype=np.float32) # cityscapes norm
    # DEEPLAB_STD  = np.array([0.18696375, 0.19017339, 0.18720214], dtype=np.float32) 
    DEEPLAB_MEAN = np.array([0.42935305, 0.42347938, 0.40977437], dtype=np.float32) # gta norm
    DEEPLAB_STD  = np.array([0.25669742, 0.25097305, 0.24708469], dtype=np.float32) 

def set_determinism():
    # set seed, to be deterministic
    seed = 123
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def tensor2imgs(tensor):
    #NOTE: tensor must be in range (0,1)
    image = tensor.detach().permute(0, 2, 3, 1).cpu().numpy()
    return (image * 255).astype("uint8")        

def test(cnet_aug,
         eval_model,
         test_dataloader,
         epoch):
    
    prompt = Config.PROMPT
    cnet_normalize = Normalize(Config.CNET_MEAN, Config.CNET_STD)
    deeplab_normalize = Normalize(Config.DEEPLAB_MEAN, Config.DEEPLAB_STD)
    cnet_bilinear_resize = lambda x: x # Resize(Config.CNET_INSHAPE, InterpolationMode.BILINEAR)
    cnet_nearest_resize = lambda x: x # Resize(Config.CNET_INSHAPE, InterpolationMode.NEAREST)    
    deeplab_bilinear_resize = lambda x: x # Resize(Config.DEEPLAB_INSHAPE, InterpolationMode.BILINEAR)        
        
    class_names = test_dataloader.dataset.CLASSES
    void_idx = test_dataloader.dataset.VOID_IDX
    
    num_classes = len(class_names)
    total_area_intersect = torch.zeros((num_classes, ), dtype=torch.float64)
    total_area_union = torch.zeros((num_classes, ), dtype=torch.float64)
    total_area_pred_label = torch.zeros((num_classes, ), dtype=torch.float64)
    total_area_label = torch.zeros((num_classes, ), dtype=torch.float64)  
    
    cnet_list = []
    rgb_list = []
    condition_list = []
    base_total_area_intersect = torch.zeros((num_classes, ), dtype=torch.float64)
    base_total_area_union = torch.zeros((num_classes, ), dtype=torch.float64)
    base_total_area_pred_label = torch.zeros((num_classes, ), dtype=torch.float64)
    base_total_area_label = torch.zeros((num_classes, ), dtype=torch.float64)  
    
    max_batches = len(test_dataloader)
    
    pbar = tqdm(total=max_batches)
    pbar.set_description(f"eval epoch {epoch}")
    for bidx, data in enumerate(test_dataloader):
        
        assert len(data) == 3
        
        image, gt, condition = data # RGB, float32, (769x769), norm [0,1]
                                                    
        condition = (condition.float() / 255)
        condition = torch.repeat_interleave(condition, 3, dim=1) # (B, 1, H, W) -> (B, 3, H, W)
                                
        image = image.to(device)
        condition = condition.to(device)
        
        # normalize [0,1]➙[-1,1]
        image_rsz = cnet_bilinear_resize(image)
        condition_rsz = cnet_nearest_resize(condition)
        image_rsz_norm = cnet_normalize(image_rsz)
                                            
        #NOTE: this expects RGB
        with torch.no_grad():
            diffusion_pred, *_ = cnet_aug(image_rsz_norm, condition_rsz, prompt)
                                
        diffusion_pred_rsz = deeplab_bilinear_resize(diffusion_pred)
        diffusion_pred_rsz_norm = deeplab_normalize(diffusion_pred_rsz)  
                                    
        with torch.no_grad():
            #NOTE: this expects RGB
            pred = eval_model(diffusion_pred_rsz_norm)["out"]        
                                
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

        # compute baselines
        with torch.no_grad():
            pred = eval_model(deeplab_normalize(image))["out"]
            
        base_pred_classes = pred.argmax(1).cpu().numpy()
            
        base_area_intersect, base_area_union, \
        base_area_pred_label, base_area_label = total_intersect_and_union(pred_classes, 
                                                                          gt, 
                                                                          num_classes, 
                                                                          void_idx, 
                                                                          label_map=None,
                                                                          reduce_zero_label=False)
        base_total_area_intersect += base_area_intersect
        base_total_area_union += base_area_union
        base_total_area_pred_label += base_area_pred_label
        base_total_area_label += base_area_label
                                                   
        # save results every N batches
        if bidx % (max(max_batches // 4, 1)) != 0: 
            pbar.update(1)
            continue
        
        diffusion_rsz_color = diffusion_pred_rsz.detach().permute(0, 2, 3, 1).cpu().numpy()
        diffusion_rsz_color = (diffusion_rsz_color * 255).astype("uint8")
                                       
        for im_color, pred_mask, true_mask in zip(diffusion_rsz_color, pred_classes, gt):
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
            
            cnet_list.append(wandb.Image(im_color, masks=masks))
            
        image_color = tensor2imgs(image) # uint8
        for im_color, pred_mask, true_mask in zip(image_color, base_pred_classes, gt):
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
            
            rgb_list.append(wandb.Image(im_color, masks=masks))

        condition_color = condition.detach().permute(0, 2, 3, 1).cpu().numpy()
        condition_color = (condition_color * 255).astype("uint8")
        condition_list.extend([wandb.Image(cond) for cond in condition])            

        pbar.update(1)
            
    pbar.close()
        
    ret_metrics = total_area_to_metrics(total_area_intersect, 
                                        total_area_union,
                                        total_area_pred_label,
                                        total_area_label, 
                                        metrics=["mIoU"], 
                                        nan_to_num=0,
                                        beta=1)       
           
    mean_accs, mean_ious = ret_metrics["Acc"], ret_metrics["IoU"]
    
    # Remove the void class from the metrics
    mean_accs = np.delete(mean_accs, void_idx)
    mean_ious = np.delete(mean_ious, void_idx)
    class_names = list(np.delete(np.array(class_names, dtype=object), void_idx))
    
    mean_acc, mean_iou = mean_accs.mean(), mean_ious.mean()
    table = wandb.Table(data=[("mAcc", mean_acc), ("mIoU", mean_iou)], columns=["metric", "score"])
    plot = wandb.plot.bar(table, label="metric", value="score", title="Mean metrics across classes")
    wandb.log({"eval/metrics/mean_plot": plot}, step=epoch)          
        
    for metric_name, metric_values in zip(["Acc", "IoU"], [mean_accs, mean_ious]):
        # Create a wandb.Table with columns for class names and metric values
        table_data = list(zip(class_names, metric_values))
        table = wandb.Table(data=table_data, columns=["class", metric_name])

        # Create a bar graph using wandb.plot.bar
        plot = wandb.plot.bar(table, label="class", value=metric_name, title=f'Per Class {metric_name}')

        # Log the table and plot with a specific topic name
        topic_name = f"eval/metrics/{metric_name.lower()}"
        wandb.log({f"{topic_name}_plot": plot}, step=epoch)          
        
    wandb.log({"eval/images/cnet": cnet_list}, step=epoch)
    
    # log baselines
    wandb.log({"eval/images/condition": condition_list}, step=epoch)            
    wandb.log({"baseline/images/raw": rgb_list}, step=epoch)
    
    base_ret_metrics = total_area_to_metrics(base_total_area_intersect, 
                                                base_total_area_union,
                                                base_total_area_pred_label,
                                                base_total_area_label, 
                                                metrics=["mIoU"], 
                                                nan_to_num=0,
                                                beta=1)       
        
    base_mean_accs = base_ret_metrics["Acc"]
    base_mean_ious = base_ret_metrics["IoU"]  
    for metric_name, metric_values in zip(["Acc", "IoU"], [base_mean_accs, base_mean_ious]):
        # Create a wandb.Table with columns for class names and metric values
        table_data = list(zip(class_names, metric_values))
        table = wandb.Table(data=table_data, columns=["class", metric_name])

        # Create a bar graph using wandb.plot.bar
        plot = wandb.plot.bar(table, label="class", value=metric_name, title=f'Per Class {metric_name}')

        # Log the table and plot with a specific topic name
        topic_name = f"baseline/metrics/{metric_name.lower()}"
        wandb.log({f"{topic_name}_plot": plot}, step=epoch)  
            
def main(args):          
    global device
    device = args.device
    
    # avoid randomicity
    set_determinism()
    
    if DatasetType[args.dataset_type] == DatasetType.CITYSCAPES:
        from deeplab.cityscapes_dataset import get_dataloader, ConditionType
    elif DatasetType[args.dataset_type] == DatasetType.KITTI:
        from deeplab.kitti_dataset import get_dataloader, ConditionType
    
    mean_and_std = ((0., 0., 0.), (1., 1., 1.)) # Set normalization in range (0,1)
    condition_type = ConditionType[args.condition_type]
    test_dataloader = get_dataloader(args.dataset_path, 
                                     split=args.dataset_split, 
                                     batch_size=args.batch_size, 
                                     nworkers=args.nworkers,
                                     filter_labels=args.filter_labels,
                                     mean_and_std=mean_and_std,
                                     condition_type=condition_type)
    
    scheduler_type = SchedulerType[args.scheduler]
    denoise_type = DenoiseType[args.denoise]
        
    augmentation_model = ControlNetAugmentation(args.cnet_ckpt, 
                                                scheduler_type=scheduler_type,
                                                eta=args.eta,
                                                max_timesteps=args.max_timesteps,
                                                num_timesteps=args.num_timesteps,
                                                denoise_type=denoise_type,
                                                device=device)
    # augmentation_model = T2IAdapterAugmentation(args.cnet_ckpt, 
    #                                             scheduler_type=scheduler_type,
    #                                             eta=args.eta,
    #                                             max_timesteps=args.max_timesteps,
    #                                             num_timesteps=args.num_timesteps,
    #                                             device=device)
    augmentation_model.set_eval()
    
    deeplab_ckpt = torch.load(args.deeplab_ckpt, map_location=device)
    # load state dict
    if isinstance(deeplab_ckpt, dict):
        num_classes = len(test_dataloader.dataset.CLASSES)
        eval_model = deeplabv3_resnet101(num_classes=num_classes)
        eval_model.load_state_dict(deeplab_ckpt["model"])
        epoch = deeplab_ckpt.get("epoch", 0)
    else:
        eval_model = deeplab_ckpt        
        epoch = 0
        
    if args.deeplab_mode == "eval":
        eval_model.eval()
    elif args.deeplab_mode == "train":
        eval_model.train()
        
    eval_model.to(device)
    
    # Enable TF32 for faster training on Ampere GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    
    wandb.init(project="Deeplabv3andCNet",
               name=f"{args.exp}_{args.dataset_type}_{args.dataset_split}",
               tags=["test"],
               dir=os.environ.get("WANDB_DIR"))
    
    test(augmentation_model,
         eval_model, 
         test_dataloader,
         epoch)
    
class DatasetType(Enum):
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
    parser.add_argument("--deeplab_ckpt", 
                        required=True,
                        type=str,
                        help="Path to DeepLabv3 pretrained weights.")   
    parser.add_argument("--deeplab_mode", 
                        type=str,
                        default="eval",
                        choices=["eval", "train"],
                        help="Which mode to use with DeepLabv3")    
    parser.add_argument("--dataset_path", 
                        required=True,
                        type=str,
                        help="Path to Cityscapes dataset.")
    parser.add_argument("--dataset_split", 
                        default="val",
                        type=str,
                        help="Which split to use.")      
    parser.add_argument("--dataset_type", 
                        type=str,
                        default=DatasetType.CITYSCAPES.name,
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
    parser.add_argument("--filter_labels", 
                        action="store_true",
                        help="Whether to remove labels with small area.")      
    parser.add_argument("--condition_type", 
                        type=str,
                        default=ConditionType.NONE.name,
                        choices=ConditionType.options(), 
                        help="Which condition to use in ControlNet.")                        
    parser.add_argument("--cnet_ckpt", 
                        default="lllyasviel/control_v11p_sd15_canny", 
                        help="ControlNet cpkt path.")   
    parser.add_argument("--max_timesteps", 
                        type=int,
                        default=1000, 
                        help="Max number of timesteps in diffusion.")   
    parser.add_argument("--num_timesteps", 
                        type=int,
                        default=20, 
                        help="How many timesteps to use, "
                             "timesteps = max_timesteps / num_timesteps.")   
    parser.add_argument("--scheduler", 
                        type=str,
                        default=SchedulerType.DDIM.name,
                        choices=SchedulerType.options(), 
                        help="Scheduler to be used.")
    parser.add_argument("--eta", 
                        default=0.0,
                        type=float,
                        help="ETA for DDIM.")
    parser.add_argument("--denoise", 
                        type=str,
                        default=DenoiseType.PARTIAL_DENOISE_T_FIXED.name,
                        choices=DenoiseType.options(), 
                        help="Type of denoise to be used.")      
    parser.add_argument("--device", 
                        type=str,
                        default="cuda:0", 
                        help="Which device to use for the training.")       
    return parser.parse_args()
    
if __name__ == "__main__":
    args = parse_args()
    main(args)
    

    