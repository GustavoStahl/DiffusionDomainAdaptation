import torch
import torch.nn.functional as F

from diffusers import ControlNetModel
from diffusers.utils.import_utils import is_xformers_available

from base_augmentation import BaseAugmentation, DenoiseType, SchedulerType

from tqdm import tqdm

from dataclasses import dataclass

from contextlib import nullcontext

@dataclass
class ControlNetBackwardsHelper(object):
    x0: torch.FloatTensor = None
    xt_prev: torch.FloatTensor = None
    noise: torch.FloatTensor = None
    timestep_xt_prev: torch.IntTensor = None
    
    def aslist(self):
        return [self.x0, self.xt_prev, self.noise, self.timestep_xt_prev]

class ControlNetAugmentation(BaseAugmentation):
    def __init__(self, 
                 model_path, 
                 scheduler_type=SchedulerType.DDIM,
                 eta=0.0,
                 max_timesteps=1000,
                 num_timesteps=20,
                 denoise_type=DenoiseType.PARTIAL_DENOISE_T_FIXED,
                 device="cuda"):
        super().__init__(scheduler_type, eta, max_timesteps, num_timesteps, device, denoise_type)
        
        controlnet = ControlNetModel.from_pretrained(model_path)
        
        if is_xformers_available():
            print("xformers available, using it optimize controlnet.")
            controlnet.enable_xformers_memory_efficient_attention()
                
        self.set_condition_model(controlnet)
        self.set_train()
        
    def partial_denoise_train(self, noisy_latents, noise, embeddings, condition, timesteps):
        down_block_res_samples, mid_block_res_sample = self.condition_model(
            noisy_latents,
            timesteps,
            encoder_hidden_states=embeddings,
            controlnet_cond=condition,
            return_dict=False,
        )

        # Predict the noise residual
        noise_pred = self.unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=embeddings,
            down_block_additional_residuals=[
                sample.to(dtype=torch.float32) for sample in down_block_res_samples
            ],
            mid_block_additional_residual=mid_block_res_sample.to(dtype=torch.float32),
        ).sample
                
        # Get the target for loss depending on the prediction type
        if self.scheduler.config.prediction_type == "epsilon":
            target = noise
        else:
            raise ValueError(f"Unknown prediction type {self.scheduler.config.prediction_type}")
        loss = F.mse_loss(noise_pred.float(), target.float(), reduction="mean")

        # perfom x0 rough prediction (xt ➙ x0 in 1 step)
        alpha_prod_t = self.scheduler.alphas_cumprod[timesteps]
        beta_prod_t = 1 - alpha_prod_t
        alpha_prod_t = alpha_prod_t[:,None,None,None].to(noisy_latents.device)
        beta_prod_t = beta_prod_t[:,None,None,None].to(noisy_latents.device)
        noisy_latents = (noisy_latents - beta_prod_t ** (0.5) * noise_pred) / alpha_prod_t ** (0.5)
        
        return noisy_latents, loss
    
    def full_denoise_train(self, noisy_latents, noise, embeddings, condition, timesteps):
        xt = noisy_latents
        
        loss = 0.0
        backwards_helper = ControlNetBackwardsHelper(noise=noise, timestep_xt_prev=timesteps[1:2])
        timesteps = tqdm(timesteps, leave=None, desc="denoising step-by-step")
        for idx, t in enumerate(timesteps):  
            
            first_pass = idx == 0
            with nullcontext() if first_pass else torch.no_grad():
                down_block_res_samples, mid_block_res_sample = self.condition_model(
                    xt,
                    t,
                    encoder_hidden_states=embeddings,
                    controlnet_cond=condition,
                    return_dict=False,
                )

                # Predict the noise residual
                noise_pred = self.unet(
                    xt,
                    t,
                    encoder_hidden_states=embeddings,
                    down_block_additional_residuals=[
                        sample.to(dtype=torch.float32) for sample in down_block_res_samples
                    ],
                    mid_block_additional_residual=mid_block_res_sample.to(dtype=torch.float32),
                ).sample

                # perfom 1 denoise step
                if self.scheduler_type == SchedulerType.DDIM:
                    xt_prev = self.scheduler.step(noise_pred, t, xt, eta=self.eta, return_dict=False)[0]
                elif self.scheduler_type == SchedulerType.DDPM:
                    xt_prev = self.scheduler.step(noise_pred, t, xt, return_dict=False)[0]
                    
                if first_pass:
                    # Get the target for loss depending on the prediction type
                    if self.scheduler.config.prediction_type == "epsilon":
                        target = noise
                    else:
                        raise ValueError(f"Unknown prediction type {self.scheduler.config.prediction_type}")
                    loss = F.mse_loss(noise_pred.float(), target.float(), reduction="mean")
                                        
                    backwards_helper.xt_prev = xt_prev
                    xt = xt_prev.detach().clone()
                    xt.requires_grad_(False)
                    self.condition_model.requires_grad_(False)
                
        x0 = xt_prev.detach().clone()
        x0.requires_grad_(True)
        backwards_helper.x0 = x0
        self.condition_model.requires_grad_(True)
            
        return x0, loss, backwards_helper
    
    def denoise_eval(self, noisy_latents, embeddings, condition, timesteps):
        timesteps = tqdm(timesteps, leave=None, desc="denoising step-by-step")
        for t in timesteps:
            #NOTE [DEBUG]: the result of this opperation occupies ~14Gb of VRAM 
            down_block_res_samples, mid_block_res_sample = self.condition_model(
                noisy_latents,
                t,
                encoder_hidden_states=embeddings,
                controlnet_cond=condition,
                return_dict=False,
            )

            # Predict the noise residual
            #NOTE [DEBUG]: the result of this opperation occupies ~53Gb of VRAM 
            noise_pred = self.unet(
                noisy_latents,
                t,
                encoder_hidden_states=embeddings,
                down_block_additional_residuals=[
                    sample.to(dtype=torch.float32) for sample in down_block_res_samples
                ],
                mid_block_additional_residual=mid_block_res_sample.to(dtype=torch.float32),
            ).sample
                    
            # perfom 1 denoise step
            if self.scheduler_type == SchedulerType.DDIM:
                noisy_latents = self.scheduler.step(noise_pred, t, noisy_latents, eta=self.eta, return_dict=False)[0]
            if self.scheduler_type == SchedulerType.DDPM:
                noisy_latents = self.scheduler.step(noise_pred, t, noisy_latents, return_dict=False)[0]
                
        return noisy_latents
                
    def forward(self, image, condition, prompt):
        #NOTE: image:     should be normalized between [-1., 1.]
        #      condition: should be normalized between [ 0., 1.]
               
        # encode prompt to vector space
        embeddings = self.encode_prompt(prompt, batch_size=len(image))
        # encode images into the latent space
        latents = self.encode_image(image)
        
        # sample noise to be added to the latents
        noise = self.get_noise(latents)
        
        timesteps = self.get_timesteps(latents)
        noisy_latents = self.add_noise(latents, noise, timesteps)
                    
        loss = 0.0
        backwards_helper = ControlNetBackwardsHelper()
        if self.is_eval:
            if self.denoise_type in DenoiseType.partial_denoise():
                #TODO change this to its own validation method
                noisy_latents, _ = self.partial_denoise_train(noisy_latents, noise, embeddings, condition, timesteps)
            elif self.denoise_type in DenoiseType.full_denoise():
                noisy_latents = self.denoise_eval(noisy_latents, embeddings, condition, timesteps)
        else:
            if self.denoise_type in DenoiseType.partial_denoise():
                noisy_latents, loss = self.partial_denoise_train(noisy_latents, noise, embeddings, condition, timesteps)
            elif self.denoise_type in DenoiseType.full_denoise():
                noisy_latents, loss, backwards_helper = self.full_denoise_train(noisy_latents, noise, embeddings, condition, timesteps)
                
        denoised_latents = noisy_latents
        
        # decode latents back to pixel space
        diffusion_pred = self.decode_image(denoised_latents)
        
        return diffusion_pred, loss, backwards_helper