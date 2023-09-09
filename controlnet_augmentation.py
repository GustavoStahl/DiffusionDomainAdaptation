import torch
from torch import nn

from transformers import CLIPTextModel, AutoTokenizer
from diffusers import (
    ControlNetModel, 
    AutoencoderKL, 
    UNet2DConditionModel,
    DDIMScheduler,
    DDPMScheduler
)

from diffusers.utils.import_utils import is_xformers_available

from tqdm import tqdm

class ControlNetAugmentation(nn.Module):
    def __init__(self, 
                 model_path, 
                 num_inference_steps=50,
                 scheduler_type="DDIM",
                 eta=0.0):
        super().__init__()
        
        device = "cuda"
        
        controlnet = ControlNetModel.from_pretrained(model_path)
        vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae")
        unet = UNet2DConditionModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="unet")
        text_encoder = CLIPTextModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="text_encoder")
        
        if scheduler_type == "DDIM":
            scheduler = DDIMScheduler.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="scheduler")
        elif scheduler_type == "DDPM":
            scheduler = DDPMScheduler.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="scheduler")
                      
        controlnet.train()    
        vae.requires_grad_(False)
        unet.requires_grad_(False)
        text_encoder.requires_grad_(False)
        
        vae.to(device)
        unet.to(device)
        controlnet.to(device)
        text_encoder.to(device)
        
        if is_xformers_available():
            print("xformers available, using it optimize unet and controlnet.")
            unet.enable_xformers_memory_efficient_attention()
            controlnet.enable_xformers_memory_efficient_attention()
                
        tokenizer = AutoTokenizer.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="tokenizer", use_fast=False)
        
        self.is_eval = False
        
        self.vae = vae
        self.unet = unet
        self.device = device
        self.tokenizer = tokenizer
        self.scheduler = scheduler
        self.controlnet = controlnet
        self.text_encoder = text_encoder
        self.num_inference_steps = num_inference_steps
        
        self.eta = eta
        self.scheduler_type = scheduler_type
        
    def forward(self, image, condition, prompt):
        #NOTE: image:     should be normalized between [-1., 1.]
        #      condition: should be normalized between [ 0., 1.]
        if isinstance(prompt, str):
            prompt = [prompt] * len(image)
        if isinstance(prompt, list):
            tokens = self.tokenizer(prompt, 
                                    max_length=self.tokenizer.model_max_length, 
                                    padding="max_length", 
                                    return_tensors="pt")
            tokens = tokens.to(self.device)
            embeddings = self.text_encoder(tokens.input_ids)[0]
        
        # encode images into the latent space
        latents = self.vae.encode(image).latent_dist.sample()
        latents *= self.vae.config.scaling_factor
        
        # sample noise to be added to the latents
        noise = torch.randn_like(latents)
        
        # Sample a random timestep for each image
        batch_size = latents.shape[0]
        if self.is_eval:
            self.scheduler.set_timesteps(self.num_inference_steps, device=latents.device)
            timesteps = self.scheduler.timesteps
            noisy_latents = self.scheduler.add_noise(latents, noise, timesteps[0].repeat(batch_size))
        else:
            #NOTE: Method 1
            # timesteps = torch.tensor([self.num_inference_steps] * batch_size, device=latents.device)
            
            #NOTE: Method 2
            #TODO investigate, should use num_inference_step or scheduler.config.num_train_timesteps
            timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, (batch_size,), device=latents.device)
            
            #NOTE: Method 3
            # self.scheduler.set_timesteps(self.num_inference_steps, device=latents.device)
            # indices = torch.randint(len(self.scheduler.timesteps), (batch_size,))
            # timesteps = self.scheduler.timesteps[indices]
            
            noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)
        
        # hacky way of combining train with validation
        if not self.is_eval:
            timesteps = timesteps[None]
            
        timesteps = tqdm(timesteps, leave=None, desc="denoising step-by-step") if self.is_eval else timesteps
        for t in timesteps:
            #NOTE [DEBUG]: the result of this opperation occupies ~14Gb of VRAM 
            down_block_res_samples, mid_block_res_sample = self.controlnet(
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
            if self.is_eval:
                if self.scheduler_type == "DDIM":
                    noisy_latents = self.scheduler.step(noise_pred, t, noisy_latents, eta=self.eta, return_dict=False)[0]
                if self.scheduler_type == "DDPM":
                    noisy_latents = self.scheduler.step(noise_pred, t, noisy_latents, return_dict=False)[0]
            # perfom x0 prediction
            else:
                alpha_prod_t = self.scheduler.alphas_cumprod[t]
                beta_prod_t = 1 - alpha_prod_t
                alpha_prod_t = alpha_prod_t[:,None,None,None].to(latents.device)
                beta_prod_t = beta_prod_t[:,None,None,None].to(latents.device)
                noisy_latents = (noisy_latents - beta_prod_t ** (0.5) * noise_pred) / alpha_prod_t ** (0.5)
                
        denoised_latents = noisy_latents
        
        denoised_latents = denoised_latents.clamp(-1., 1.)
                        
        # decode latents back to pixel space
        denoised_latents /= self.vae.config.scaling_factor
        diffusion_pred = self.vae.decode(denoised_latents).sample
        diffusion_pred = (diffusion_pred / 2 + 0.5).clamp(0, 1)
        
        return diffusion_pred
    
    def get_trainable_params(self):
        return self.controlnet.parameters()
    
    def save_weights(self, path):
        self.controlnet.save_pretrained(path, safe_serialization=True)
           
    def set_train(self):
        self.is_eval = False
        self.controlnet.train()
        
    def set_eval(self):
        self.is_eval = True
        self.controlnet.eval()