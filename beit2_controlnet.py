# core libraries
import os
import random
import gc

# computer vision and arrays
import cv2
import numpy as np

# deep learning
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.transforms import Normalize, Resize, InterpolationMode
from mmcv.image import tensor2imgs
from mmseg.core import eval_metrics
from pytorch_msssim import MS_SSIM

# custom scripts
from controlnet_augmentation import ControlNetAugmentation
from t2i_augmentation import T2IAdapterAugmentation
from base_augmentation import SchedulerType, DenoiseType
#NOTE: for this import to work you should add BEIT's directory path to 
#      the envionment variable PYTHONPATH, doing with sys.path doesn't work
#      e.g., PYTHONPATH=$PYTHONPATH:$PWD/beit2 python file.py
from beit2.tools import openmm_utils as ommutils

# visualization
import wandb
import argparse
from tqdm import tqdm

# torchshow for debugging
import torchshow as ts
# necessary for torchshow
import matplotlib; matplotlib.use("Agg")

from dataclasses import dataclass

device = None

@dataclass
class Config(object):
    #NOTE hardcoded prompt
    PROMPT = ""
    # input shape
    CNET_INSHAPE = (512,512)
    BEIT_INSHAPE = (224,224)
    # normalization values
    CNET_MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32) # convert [0,1]➙[-1,1]
    CNET_STD  = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    BEIT_MEAN = np.array([0.3924615, 0.3799202, 0.3638257], dtype=np.float32) #NOTE: mean and std taken 
    BEIT_STD  = np.array([0.2962988, 0.2934247, 0.2939357], dtype=np.float32) #      from configs/gustavo_sign.py

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

def test(cnet_aug,
         eval_model,
         test_dataloader,
         epoch,
         max_batches=None):
    
    prompt = Config.PROMPT
    cnet_normalize = Normalize(Config.CNET_MEAN, Config.CNET_STD)
    beit_normalize = Normalize(Config.BEIT_MEAN, Config.BEIT_STD)
    beit_bilinear_resize = Resize(Config.BEIT_INSHAPE, InterpolationMode.BILINEAR)
        
    class_names = test_dataloader.dataset.CLASSES
    
    mean_iou = np.zeros(len(class_names), dtype=np.float32)
    mean_acc = np.zeros(len(class_names), dtype=np.float32)
    
    first_epoch = epoch == 0
    
    cnet_list = []
    if first_epoch:
        rgb_list = []
        canny_list = []
        base_mean_iou = np.zeros(len(class_names), dtype=np.float32)
        base_mean_acc = np.zeros(len(class_names), dtype=np.float32)
    
    max_batches = len(test_dataloader) if max_batches is None else max_batches
    
    pbar = tqdm(total=max_batches)
    pbar.set_description(f"eval epoch {epoch}")
    for bidx, data in enumerate(test_dataloader):
        
        assert len(data) == 3
        
        # garbage collector
        clear_cache()       
        
        image = data["img"].data[0] # RGB, float32, (512, 512), norm [0,1]
        metas = data["img_metas"].data[0]
        gt    = data["gt_semantic_seg"].data[0]
                                            
        image_color = tensor2imgs(image, **metas[0]["img_norm_cfg"]) # RGB➙BGR, uint8
        
        canny_guide = get_canny(image_color)
        canny_guide = torch.from_numpy(canny_guide)[:,None]
        canny_guide = torch.repeat_interleave(canny_guide, 3, dim=1) # (B, 1, H, W) -> (B, 3, H, W)
                                
        image = image.to(device)
        canny_guide = canny_guide.to(device)
        
        # normalize [0,1]➙[-1,1]
        image_norm = cnet_normalize(image)
                                            
        #NOTE: this expects RGB
        with torch.no_grad():
            diffusion_pred, *_ = cnet_aug(image_norm, canny_guide, prompt)
                                
        diffusion_pred_rsz = beit_bilinear_resize(diffusion_pred)
        diffusion_pred_norm = beit_normalize(diffusion_pred_rsz)  
        
        # replace mean and std by the ones used in beit pretrained
        for meta in metas:
            meta["img_norm_cfg"]["mean"] = Config.BEIT_MEAN * 255.
            meta["img_norm_cfg"]["std"]  = Config.BEIT_STD * 255.

        diffusion_color = diffusion_pred.detach().permute(0, 2, 3, 1).cpu().numpy()
        diffusion_color = (diffusion_color * 255).astype("uint8")
                        
        with torch.no_grad():
            eval_model_pred = eval_model(img=[diffusion_pred_norm], 
                                         img_metas=[metas], 
                                         return_loss=False)
                                
        gt_numpy = gt.permute(0, 2, 3, 1).numpy().squeeze()
        
        ret_metrics = eval_metrics(eval_model_pred,
                                   gt_numpy,
                                   len(class_names),
                                   ignore_index=0,
                                   metrics=["mIoU"],
                                   label_map=None)
        
        mean_acc += np.nan_to_num(ret_metrics["Acc"])
        mean_iou += np.nan_to_num(ret_metrics["IoU"])
        
        # compute baselines
        if first_epoch:
            with torch.no_grad():
                eval_model_pred = eval_model(img=[beit_normalize(beit_bilinear_resize(image))], 
                                             img_metas=[metas], 
                                             return_loss=False)
                
            base_metrics = eval_metrics(eval_model_pred,
                                        gt_numpy,
                                        len(class_names),
                                        0,
                                        ["mIoU"],
                                        label_map=None,
                                        reduce_zero_label=False)
            base_mean_acc += np.nan_to_num(base_metrics["Acc"])
            base_mean_iou += np.nan_to_num(base_metrics["IoU"]) 
                                                   
        # save results every N batches
        if bidx % (max(max_batches // 4, 1)) != 0: 
            clear_cache()
            pbar.update(1)
            if bidx + 1 == max_batches:
                break
            continue
                                       
        for im_color, pred_mask, true_mask in zip(diffusion_color, eval_model_pred, gt_numpy):
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
        
    wandb.log({"eval/images/cnet": cnet_list}, step=epoch)
    # log baselines
    if first_epoch:
        wandb.log({"eval/images/canny": canny_list}, step=epoch)            
        wandb.log({"baseline/images/raw": rgb_list}, step=epoch)
        
        base_mean_acc /= max_batches
        base_mean_iou /= max_batches
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
          optimizer,
          epoch,
          diffusion_loss_alpha,
          max_batches=None):
                                
    prompt = Config.PROMPT
    cnet_normalize = Normalize(Config.CNET_MEAN, Config.CNET_STD)
    beit_normalize = Normalize(Config.BEIT_MEAN, Config.BEIT_STD)
    beit_nearest_resize = Resize(Config.BEIT_INSHAPE, InterpolationMode.NEAREST)
    beit_bilinear_resize = Resize(Config.BEIT_INSHAPE, InterpolationMode.BILINEAR)
        
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
        
        assert len(data) == 3

        # garbage collector
        clear_cache()      
        
        image = data["img"].data[0] # RGB, float32, (512, 512), norm [0,1]
        metas = data["img_metas"].data[0]
        gt    = data["gt_semantic_seg"].data[0]
                                            
        image_color = tensor2imgs(image, **metas[0]["img_norm_cfg"]) # RGB➙BGR, uint8
        
        canny_guide = get_canny(image_color)
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
                                
        gt_rsz = beit_nearest_resize(gt) # nearest neighbors is needed to preserve the labels
        diffusion_pred_rsz = beit_bilinear_resize(diffusion_pred)
        diffusion_pred_norm = beit_normalize(diffusion_pred_rsz)  
        
        # replace mean and std by the ones used in beit pretrained
        for meta in metas:
            meta["img_norm_cfg"]["mean"] = Config.BEIT_MEAN * 255.
            meta["img_norm_cfg"]["std"]  = Config.BEIT_STD * 255.
            
        #NOTE: this expects RGB
        beit_logs = eval_model(gt_semantic_seg=gt_rsz, 
                               img=diffusion_pred_norm, 
                               img_metas=metas)
        
        beit_acc = {k:v for (k,v) in beit_logs.items() if "acc" in k}
        beit_losses = {k:v for (k,v) in beit_logs.items() if "loss" in k}
        
        for name, metric in beit_acc.items():
            accum_metric(metrics_tracker, f"metrics/{name}", metric.item())
            
        for name, loss in beit_losses.items():
            accum_metric(metrics_tracker, f"losses/{name}", loss.item())
                        
        total_loss = sum(beit_losses.values())
                
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
         optimizer,
         lr_scheduler,
         epochs,
         diffusion_loss_alpha,
         max_batches=None,
         no_eval=False):
    
    best_score = 0.
    for epoch in range(epochs):
        cnet_aug.set_train()
        train(cnet_aug, eval_model, train_dataloader, optimizer, epoch, diffusion_loss_alpha, max_batches=max_batches)
        if epoch % 10 != 0 or no_eval:
            continue
        cnet_aug.set_eval()
        acc, iou = test(cnet_aug, eval_model, test_dataloader, epoch, max_batches=max_batches)
        eval_score = (acc + iou) / 2.
        lr_scheduler.step(eval_score)
        if eval_score > best_score:
            print("[INFO] Better weights found! Saving them...")
            cnet_aug.save_weights(os.path.join(wandb.run.dir, "weights"))
            
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp", 
                        default=None, required=False, 
                        help="Experiment name.")
    parser.add_argument("-c", "--config", 
                        required=True, 
                        help="MMSeg config file path.")
    parser.add_argument("--beit_ckpt", 
                        required=True, 
                        help="BEITv2 cpkt path.")  
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

def main(args):
    #TODO  4. LPIPS?
    #TODO  5. update fewer parameters from controlnet
    #TODO  6. add a regularizer, L1 or L2 loss between a locked controlnet and our treinable one
    #TODO  7. change from DDPM to DDIM (test inversion)
    #NOTE: DDPM adds a lot of stochasticity, hence the image generated should look more unfamiliar than
    #      the original. Of course, we want change the image, but we don't want to deviate so much from 
    #      the input. That said, using DDIM we can somewhat be faithful to the input while getting the 
    #      randomicity/divergence from controlnet injections
    #TODO  8. remove the variation of denoising steps, keep it fixed to a low number of timesteps (e.g. 100) (done)
    #NOTE: we assume that SD has learned everything we want from the world, so no need to keep tuning
    #      different denoising steps. Therefore, we can fix the timesteps.
    #TODO  9. denoise using SD until halfway, then denoise using SD+CNET
    #TODO 10. denoise step by step, rather than in one pass
    #TODO 11. add LORA to ControlNet
    #TODO 12. check synthetic data validation results for BEiTv2 trained also on synthetic 
    #TODO 13. check for networks trained on GTA. Useful for having baselines.
           
    global device
    device = args.device
    cfg_path = args.config
    
    # avoid randomicity
    set_determinism()
    
    train_dataloader = ommutils.get_dataloader(cfg_path, split="train", batch_size=args.batch_size)
    test_dataloader = ommutils.get_dataloader(cfg_path, split="test", batch_size=args.batch_size)
    
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
        
    eval_model = ommutils.get_model(cfg_path, args.beit_ckpt, device=device)
    
    optimizer = AdamW(augmentation_model.get_trainable_params(), lr=args.lr)
    lr_scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, verbose=True) # new_lr = lr * factor
        
    # Enable TF32 for faster training on Ampere GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    
    wandb.init(project="BEITandCNet",
               name=args.exp)   
    wandb.run.log_code(".")
    
    loop(augmentation_model,
         eval_model, 
         train_dataloader, 
         test_dataloader,
         optimizer,
         lr_scheduler,
         args.epochs,
         args.diffusion_loss_alpha,
         args.max_batches,
         args.no_eval)
    
if __name__ == "__main__":
    args = parse_args()
    main(args)
    

    