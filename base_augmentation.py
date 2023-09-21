import torch
from torch import nn

from transformers import CLIPTextModel, AutoTokenizer

from diffusers import (
    AutoencoderKL, 
    UNet2DConditionModel,
    DDIMScheduler,
    DDPMScheduler
)

from diffusers.utils.import_utils import is_xformers_available

from enum import Enum, auto
class SchedulerType(Enum):
    DDIM = auto()
    DDPM = auto()
    
    @classmethod
    def options(self):
        return [var.name for var in list(self)]
    
class DenoiseType(Enum):
    PARTIAL_DENOISE_T_FIXED = auto()
    PARTIAL_DENOISE_T_VARIABLE = auto()
    FULL_DENOISE_T_FIXED = auto()
    FULL_DENOISE_T_VARIABLE = auto()
    
    @classmethod
    def options(self):
        return [var.name for var in list(self)]    
    
    @classmethod
    def partial_denoise(cls):
        return [cls.PARTIAL_DENOISE_T_FIXED, cls.PARTIAL_DENOISE_T_VARIABLE]
    
    @classmethod
    def full_denoise(cls):
        return [cls.FULL_DENOISE_T_FIXED, cls.FULL_DENOISE_T_VARIABLE]    
    
class BaseAugmentation(nn.Module):
    def __init__(self, 
                 scheduler_type=SchedulerType.DDIM,
                 eta=0.0,
                 max_timesteps=1000,
                 num_timesteps=20,
                 device="cuda",
                 denoise_type=DenoiseType.PARTIAL_DENOISE_T_FIXED):
        
        super().__init__()
               
        vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae")
        unet = UNet2DConditionModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="unet")
        text_encoder = CLIPTextModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="text_encoder")
        
        if scheduler_type == SchedulerType.DDIM:
            scheduler = DDIMScheduler.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="scheduler")
        elif scheduler_type == SchedulerType.DDPM:
            scheduler = DDPMScheduler.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="scheduler")
            
        if is_xformers_available():
            print("xformers available, using it optimize unet.")
            unet.enable_xformers_memory_efficient_attention()
            
        unet.requires_grad_(False)
        vae.requires_grad_(False)
        text_encoder.requires_grad_(False)
        
        tokenizer = AutoTokenizer.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="tokenizer", use_fast=False)
        
        vae.to(device)
        unet.to(device)
        text_encoder.to(device)
        
        self.vae = vae
        self.unet = unet
        self.device = device
        self.scheduler = scheduler
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        
        self.eta = eta
        self.denoise_type = denoise_type
        self.max_timesteps = max_timesteps
        self.num_timesteps = num_timesteps
        self.scheduler_type = scheduler_type
        
    def set_condition_model(self, model):
        model.to(self.device)
        self.condition_model = model
        
    def get_trainable_params(self):
        return self.condition_model.parameters()
    
    def save_weights(self, path):
        self.condition_model.save_pretrained(path, safe_serialization=True)
           
    def set_train(self):
        self.is_eval = False
        self.condition_model.train()
        
    def set_eval(self):
        self.is_eval = True
        self.condition_model.eval()   
        
    def encode_prompt(self, prompt, batch_size=None):
        if isinstance(prompt, str):
            prompt = [prompt] * batch_size
        if isinstance(prompt, list):
            tokens = self.tokenizer(prompt, 
                                    max_length=self.tokenizer.model_max_length, 
                                    padding="max_length", 
                                    return_tensors="pt")
            tokens = tokens.to(self.device)
            embeddings = self.text_encoder(tokens.input_ids)[0]
        return embeddings
    
    def encode_image(self, image):
        latents = self.vae.encode(image).latent_dist.sample()
        latents *= self.vae.config.scaling_factor
        return latents
    
    def decode_image(self, latent):
        latent = latent.clamp(-1., 1.)
                        
        # decode latents back to pixel space
        latent /= self.vae.config.scaling_factor
        latent = self.vae.decode(latent).sample
        latent = (latent / 2 + 0.5).clamp(0, 1)
        return latent
    
    def get_noise(self, latents):
        return torch.randn_like(latents)
    
    def get_timesteps(self, latents):
        # Sample a random timestep for each image
        
        if self.is_eval:
            return self.get_timesteps_eval(latents)
        else:
            return self.get_timesteps_train(latents)
            
    def get_timesteps_train(self, latents):
        batch_size = len(latents)
        
        #NOTE: Initial timesteps changes the texture, we are after them.
        
        if self.denoise_type == DenoiseType.PARTIAL_DENOISE_T_FIXED:
            timesteps = torch.tensor([self.max_timesteps] * batch_size, device=latents.device)
        
        elif self.denoise_type == DenoiseType.PARTIAL_DENOISE_T_VARIABLE:
            self.scheduler.config.num_train_timesteps = self.max_timesteps
            self.scheduler.set_timesteps(self.num_timesteps, device=latents.device)
            indices = torch.randint(len(self.scheduler.timesteps), (batch_size,))
            timesteps = self.scheduler.timesteps[indices] 
            
        elif self.denoise_type == DenoiseType.FULL_DENOISE_T_FIXED:
            self.scheduler.config.num_train_timesteps = self.max_timesteps
            self.scheduler.set_timesteps(self.num_timesteps if self.num_timesteps <= self.max_timesteps else self.max_timesteps, 
                                         device=latents.device)
            timesteps = self.scheduler.timesteps            
        
        elif self.denoise_type == DenoiseType.FULL_DENOISE_T_VARIABLE:
            cur_max_t = int(torch.randint(1, self.max_timesteps, (1,)))
            self.scheduler.config.num_train_timesteps = cur_max_t
            self.scheduler.set_timesteps(self.num_timesteps if self.num_timesteps <= cur_max_t else cur_max_t, 
                                         device=latents.device)
            timesteps = self.scheduler.timesteps
        
        return timesteps
        
    def get_timesteps_eval(self, latents):
        batch_size = len(latents)
        
        #NOTE: Initial timesteps changes the texture, we are after them.
        
        if self.denoise_type in DenoiseType.partial_denoise():
            timesteps = torch.tensor([self.max_timesteps] * batch_size, device=latents.device)
        
        elif self.denoise_type in DenoiseType.full_denoise():
            self.scheduler.config.num_train_timesteps = self.max_timesteps
            self.scheduler.set_timesteps(self.num_timesteps, device=latents.device)
            timesteps = self.scheduler.timesteps
            
        return timesteps
        
    def add_noise(self, latents, noise, timesteps):
        if self.denoise_type in DenoiseType.partial_denoise():
            return self.scheduler.add_noise(latents, noise, timesteps)
        elif self.denoise_type in DenoiseType.full_denoise():
            return self.scheduler.add_noise(latents, noise, timesteps[0].repeat(len(latents)))