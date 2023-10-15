# core libraries
import os
import random
import gc

# computer vision and arrays
import cv2
import numpy as np

# deep learning
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.transforms import Normalize, Resize, InterpolationMode
from mmseg.core.evaluation.metrics import total_intersect_and_union, total_area_to_metrics

from pytorch_msssim import MS_SSIM

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
from deeplab.cityscapes_dataset import get_dataloader

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
    
def clear_cache():
    gc.collect()
    torch.cuda.empty_cache()  
    
def get_canny(image):
    # Get canny edges from input image
    canny_guide = np.empty((len(image), *image[0].shape[:2]), dtype=np.float32)
    for idx, im_color in enumerate(image):
        #NOTE: this expects BGR
        canny = cv2.Canny(im_color, 50, 100)
        # normalize between [0., 1.]
        canny_norm = canny.astype("float32") / 255.
        canny_guide[idx] = canny_norm
    return canny_guide

def tensor2imgs(tensor):
    #NOTE: tensor must be in range (0,1)
    image = tensor.detach().permute(0, 2, 3, 1).cpu().numpy()
    return (image * 255).astype("uint8")        

def test(cnet_aug,
         eval_model,
         test_dataloader,
         epoch,
         max_batches=None):
    
    prompt = Config.PROMPT
    cnet_normalize = Normalize(Config.CNET_MEAN, Config.CNET_STD)
    deeplab_normalize = Normalize(Config.DEEPLAB_MEAN, Config.DEEPLAB_STD)
        
    class_names = test_dataloader.dataset.CLASSES
    void_idx = test_dataloader.dataset.VOID_IDX
    
    num_classes = len(class_names)
    total_area_intersect = torch.zeros((num_classes, ), dtype=torch.float64)
    total_area_union = torch.zeros((num_classes, ), dtype=torch.float64)
    total_area_pred_label = torch.zeros((num_classes, ), dtype=torch.float64)
    total_area_label = torch.zeros((num_classes, ), dtype=torch.float64)  
    
    first_epoch = epoch == 0
    
    cnet_list = []
    if first_epoch:
        rgb_list = []
        canny_list = []
        base_total_area_intersect = torch.zeros((num_classes, ), dtype=torch.float64)
        base_total_area_union = torch.zeros((num_classes, ), dtype=torch.float64)
        base_total_area_pred_label = torch.zeros((num_classes, ), dtype=torch.float64)
        base_total_area_label = torch.zeros((num_classes, ), dtype=torch.float64)  
    
    max_batches = len(test_dataloader) if max_batches is None else max_batches
    
    pbar = tqdm(total=max_batches)
    pbar.set_description(f"eval epoch {epoch}")
    for bidx, data in enumerate(test_dataloader):
        
        assert len(data) == 3
        
        # garbage collector
        clear_cache()       
        
        image, gt = data # RGB, float32, (769x769), norm [0,1]
                                            
        image_color = tensor2imgs(image) # uint8
        
        image_color_bgr = image_color[...,::-1] # RGB➙BGR
        canny_guide = get_canny(image_color_bgr)
        canny_guide = torch.from_numpy(canny_guide)[:,None]
        canny_guide = torch.repeat_interleave(canny_guide, 3, dim=1) # (B, 1, H, W) -> (B, 3, H, W)
                                
        image = image.to(device)
        canny_guide = canny_guide.to(device)
        
        # normalize [0,1]➙[-1,1]
        image_norm = cnet_normalize(image)
                                            
        #NOTE: this expects RGB
        with torch.no_grad():
            diffusion_pred, *_ = cnet_aug(image_norm, canny_guide, prompt)
                                
        diffusion_pred_norm = deeplab_normalize(diffusion_pred)  
            
        diffusion_color = diffusion_pred.detach().permute(0, 2, 3, 1).cpu().numpy()
        diffusion_color = (diffusion_color * 255).astype("uint8")
                        
        with torch.no_grad():
            #NOTE: this expects RGB
            pred = eval_model(diffusion_pred_norm)["out"]
                                
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
        if first_epoch:
            with torch.no_grad():
                eval_model_pred = eval_model([deeplab_normalize(deeplab_bilinear_resize(image))])["out"]
                
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
            clear_cache()
            pbar.update(1)
            if bidx + 1 == max_batches:
                break
            continue
                                       
        for im_color, pred_mask, true_mask in zip(diffusion_color, eval_model_pred, gt):
            class_labels = {i:c for (i,c) in enumerate(class_names)}
            
            true_mask = cv2.resize(true_mask.astype("uint8"), Config.CNET_INSHAPE, interpolation=cv2.INTER_LINEAR)
            pred_mask = cv2.resize(pred_mask.astype("uint8"), Config.CNET_INSHAPE, interpolation=cv2.INTER_LINEAR)
            
            masks = {"prediction": {"mask_data": pred_mask, "class_labels": class_labels}, 
                     "ground_truth": {"mask_data": true_mask, "class_labels": class_labels}}
            
            cnet_list.append(wandb.Image(im_color, masks=masks))
            
        if first_epoch:
            for im_color, pred_mask, true_mask in zip(image_color, eval_model_pred, gt_numpy):
                class_labels = {i:c for (i,c) in enumerate(class_names)}
                
                true_mask = cv2.resize(true_mask.astype("uint8"), Config.CNET_INSHAPE, interpolation=cv2.INTER_LINEAR)
                pred_mask = cv2.resize(pred_mask.astype("uint8"), Config.CNET_INSHAPE, interpolation=cv2.INTER_LINEAR)
                
                masks = {"prediction": {"mask_data": pred_mask, "class_labels": class_labels}, 
                         "ground_truth": {"mask_data": true_mask, "class_labels": class_labels}}
                
                rgb_list.append(wandb.Image(im_color[...,::-1], masks=masks))

            canny_color = canny_guide.detach().permute(0, 2, 3, 1).cpu().numpy()
            canny_color = (canny_color[...,0] * 255).astype("uint8")
            canny_list.extend([wandb.Image(canny) for canny in canny_guide])            

        clear_cache()
        pbar.update(1)
        if bidx + 1 == max_batches:
            break
            
    pbar.close()
        
    ret_metrics = total_area_to_metrics(total_area_intersect, 
                                        total_area_union,
                                        total_area_pred_label,
                                        total_area_label, 
                                        metrics=["mIoU"], 
                                        nan_to_num=0,
                                        beta=1)       
           
    mean_acc = ret_metrics["Acc"]
    mean_iou = ret_metrics["IoU"]  
    
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
        
    wandb.log({"eval/images/cnet": cnet_list}, step=epoch)
    # log baselines
    if first_epoch:
        wandb.log({"eval/images/canny": canny_list}, step=epoch)            
        wandb.log({"baseline/images/raw": rgb_list}, step=epoch)
        
        base_ret_metrics = total_area_to_metrics(base_total_area_intersect, 
                                                 base_total_area_union,
                                                 base_total_area_pred_label,
                                                 base_total_area_label, 
                                                 metrics=["mIoU"], 
                                                 nan_to_num=0,
                                                 beta=1)       
            
        base_mean_acc = base_ret_metrics["Acc"]
        base_mean_iou = base_ret_metrics["IoU"]  
        for metric_name, metric_values in zip(["Acc", "IoU"], [base_mean_acc, base_mean_iou]):
            # Create a wandb.Table with columns for class names and metric values
            table_data = list(zip(class_names, metric_values))
            table = wandb.Table(data=table_data, columns=["class", metric_name])

            # Create a bar graph using wandb.plot.bar
            plot = wandb.plot.bar(table, label="class", value=metric_name, title=f'Per Class {metric_name}')

            # Log the table and plot with a specific topic name
            topic_name = f"baseline/metrics/{metric_name.lower()}"
            wandb.log({f"{topic_name}_plot": plot}, step=epoch)  
            
    return mean_acc.mean(), mean_iou.mean()
           
def train(cnet_aug,
          eval_model, 
          train_dataloader, 
          criterion,
          optimizer,
          epoch,
          diffusion_loss_alpha,
          max_batches=None):
                                
    prompt = Config.PROMPT
    cnet_normalize = Normalize(Config.CNET_MEAN, Config.CNET_STD)
    deeplab_normalize = Normalize(Config.DEEPLAB_MEAN, Config.DEEPLAB_STD)
        
    use_ssim = False
    if use_ssim:
        # get max window size
        win_size = (Config.CNET_INSHAPE[0] + 2**4) / 2**4 - 2 
        # 1. maximum similarity, 0. no similarity
        ms_ssim = MS_SSIM(win_size=win_size, data_range=1., size_average=True, channel=3) 
        # 0. maximum similarity, 1. no similarity
        similarity_loss_func = lambda x, y: 1.0 - ms_ssim(x,y) 
                
    metrics_tracker = {}
    def accum_metric(tracker, key, val):
        if key not in tracker:
            metrics_tracker[key] = val
        else:
            metrics_tracker[key] += val        
    
    max_batches = len(train_dataloader) if max_batches is None else max_batches
    
    pbar = tqdm(total=max_batches)
    pbar.set_description(f"train epoch {epoch}")
    for bidx, data in enumerate(train_dataloader):
        
        assert len(data) == 2

        # garbage collector
        clear_cache()  
        
        #TODO 1. resize dataset for the resoltuion of 512 x 960
        #TODO 2. normalize dataset for mean 0.5 and std 0.5
        #TODO 3. normalize controlnet output by GTA dataset's mean and std
        
        image, gt = data # RGB, float32, (769x769), norm [0,1]
        
        image_color_bgr = tensor2imgs(image)[...,::-1] # RGB➙BGR, uint8
        
        canny_guide = get_canny(image_color_bgr)
        canny_guide = torch.from_numpy(canny_guide)[:,None]
        canny_guide = torch.repeat_interleave(canny_guide, 3, dim=1) # (B, 1, H, W) -> (B, 3, H, W)
                                
        gt = gt.to(device)
        image = image.to(device)
        canny_guide = canny_guide.to(device)
        
        # normalize [0,1]➙[-1,1]
        image_norm = cnet_normalize(image)
                                            
        #NOTE: this expects RGB
        diffusion_pred, denoise_loss, backward_helper = cnet_aug(image_norm, canny_guide, prompt)
        # ts.save(diffusion_pred, f"results/{cnet_aug.scheduler.__class__.__name__}_{cnet_aug.num_inference_steps}.png")
        # import sys; sys.exit("Finishing...")
                                
        diffusion_pred_norm = deeplab_normalize(diffusion_pred)  
            
        #NOTE: this expects RGB
        pred = eval_model(diffusion_pred_norm)["out"]
        
        ce_loss = criterion(pred, gt.long())
        accum_metric(metrics_tracker, "losses/cross_entropy_loss", ce_loss.item())
                        
        total_loss = ce_loss
                
        if use_ssim:
            similarity_loss = similarity_loss_func(diffusion_pred, image)
            accum_metric(metrics_tracker, "losses/similarity_loss", similarity_loss.item())
            total_loss += similarity_loss
                                                
        optimizer.zero_grad(set_to_none=True)
        total_loss.backward(retain_graph=False) # free the vae and the segmenter
        
        if diffusion_loss_alpha > 0.0:
            accum_metric(metrics_tracker, f"losses/denoise_loss", denoise_loss.item())
            accum_metric(metrics_tracker, "losses/total_loss", total_loss.item() + denoise_loss.item())
            (diffusion_loss_alpha * denoise_loss).backward(retain_graph=True)
        else:
            accum_metric(metrics_tracker, "losses/total_loss", total_loss.item())
        
        if all(list(map(lambda x: x is not None, backward_helper.aslist()))):
            x0, xt_prev, noise, timestep_xt_prev = backward_helper.aslist()
            # xt_prev.backward(gradient=cnet_aug.add_noise(x0.grad, noise, timestep_xt_prev))
            xt_prev.backward(gradient=x0.grad)
        optimizer.step()
        
        clear_cache()       
        pbar.update(1)
        
        # process first N batches
        if bidx + 1 == max_batches:
            break

    for key, val in metrics_tracker.items():
        val_mean = val / max_batches
        wandb.log({f"train/{key}" : val_mean }, step=epoch)
        
    pbar.close()
        
def loop(cnet_aug,
         eval_model, 
         train_dataloader, 
         test_dataloader, 
         criterion,
         optimizer,
         lr_scheduler,
         epochs,
         diffusion_loss_alpha,
         max_batches=None,
         no_eval=False):
    
    best_score = 0.
    for epoch in range(epochs):
        eval_model.train()
        cnet_aug.set_train()
        train(cnet_aug, eval_model, train_dataloader, criterion, optimizer, epoch, diffusion_loss_alpha, max_batches=max_batches)
        if epoch % 10 != 0 or no_eval:
            continue
        eval_model.eval()
        cnet_aug.set_eval()
        acc, iou = test(cnet_aug, eval_model, test_dataloader, epoch, max_batches=max_batches)
        eval_score = (acc + iou) / 2.
        lr_scheduler.step(eval_score)
        if eval_score > best_score:
            print("[INFO] Better weights found! Saving them...")
            cnet_aug.save_weights(os.path.join(wandb.run.dir, "weights"))

def main(args):          
    global device
    device = args.device
    
    # avoid randomicity
    set_determinism()
    
    mean_and_std = ((0., 0., 0.), (1., 1., 1.)) # Set normalization in range (0,1)
    train_dataloader = get_dataloader(args.dataset_path, 
                                      split="train", 
                                      batch_size=args.batch_size,
                                      nworkers=args.nworkers,
                                      filter_labels=args.filter_labels,
                                      mean_and_std=mean_and_std)
    test_dataloader = get_dataloader(args.dataset_path, 
                                     split="val", 
                                     batch_size=args.batch_size, 
                                     nworkers=args.nworkers,
                                     filter_labels=args.filter_labels,
                                     mean_and_std=mean_and_std)
    
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
    
    deeplab_ckpt = torch.load(args.deeplab_ckpt, map_location=device)
    # load state dict
    if isinstance(deeplab_ckpt, dict):
        num_classes = len(test_dataloader.dataset.CLASSES)
        eval_model = deeplabv3_resnet101(num_classes=num_classes)
        eval_model.load_state_dict(deeplab_ckpt["model"])
    else:
        eval_model = deeplab_ckpt
        
    eval_model.to(device)
        
    optimizer = AdamW(augmentation_model.get_trainable_params(), lr=args.lr)
    lr_scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, verbose=True) # new_lr = lr * factor
    
    class_weights = torch.tensor(train_dataloader.dataset.CLASS_WEIGHTS, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
        
    # Enable TF32 for faster training on Ampere GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    
    wandb.init(project="Deeplabv3andCNet",
               name=args.exp)   
    wandb.run.log_code(".")
    
    loop(augmentation_model,
         eval_model, 
         train_dataloader, 
         test_dataloader,
         criterion,
         optimizer,
         lr_scheduler,
         args.epochs,
         args.diffusion_loss_alpha,
         args.max_batches,
         args.no_eval)
            
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp", 
                        default=None, required=False, 
                        help="Experiment name.")
    parser.add_argument("--deeplab_ckpt", 
                        required=True,
                        type=str,
                        help="Path to DeepLabv3 pretrained weights.")   
    parser.add_argument("--dataset_path", 
                        required=True,
                        type=str,
                        help="Path to Cityscapes dataset.")
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
    parser.add_argument("--filter_labels", 
                        action="store_true",
                        help="Whether to remove labels with small area.")                             
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
    parser.add_argument("--no_eval", 
                        action="store_true",
                        help="Don't perform validation steps")
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
    parser.add_argument("--diffusion_loss_alpha", 
                        type=float,
                        default=0.0,
                        help="Weight factor for the diffusion loss.")        
    parser.add_argument("--device", 
                        type=str,
                        default="cuda:0", 
                        help="Which device to use for the training.")       
    return parser.parse_args()
    
if __name__ == "__main__":
    args = parse_args()
    main(args)
    

    