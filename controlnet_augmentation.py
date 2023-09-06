import torch
from torch import nn

from transformers import CLIPTextModel, AutoTokenizer
from diffusers import (
    ControlNetModel, 
    AutoencoderKL, 
    UNet2DConditionModel,
    DDIMScheduler
)

from diffusers.utils.import_utils import is_xformers_available

class ControlNetAugmentation(nn.Module):
    def __init__(self, 
                 model_path, 
                 num_inference_steps=None):
        super().__init__()
        
        device = "cuda"
        
        controlnet = ControlNetModel.from_pretrained(model_path)
        vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae")
        unet = UNet2DConditionModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="unet")
        scheduler = DDIMScheduler.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="scheduler")
        text_encoder = CLIPTextModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="text_encoder")
                      
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
        
        self.vae = vae
        self.unet = unet
        self.device = device
        self.tokenizer = tokenizer
        self.scheduler = scheduler
        self.controlnet = controlnet
        self.text_encoder = text_encoder
        self.num_inference_steps = num_inference_steps
        
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
        if self.num_inference_steps is None:
            timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, (batch_size,), device=latents.device)
        else:
            timesteps = torch.tensor([self.num_inference_steps] * batch_size, device=latents.device)

        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)
                
        #NOTE [DEBUG]: the result of this opperation occupies ~14Gb of VRAM 
        down_block_res_samples, mid_block_res_sample = self.controlnet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=embeddings,
            controlnet_cond=condition,
            return_dict=False,
        )

        # Predict the noise residual
        #NOTE [DEBUG]: the result of this opperation occupies ~53Gb of VRAM 
        noise_pred = self.unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=embeddings,
            down_block_additional_residuals=[
                sample.to(dtype=torch.float32) for sample in down_block_res_samples
            ],
            mid_block_additional_residual=mid_block_res_sample.to(dtype=torch.float32),
        ).sample
                
        alpha_prod_t = self.scheduler.alphas_cumprod[timesteps]
        beta_prod_t = 1 - alpha_prod_t
        alpha_prod_t = alpha_prod_t[:,None,None,None].to(latents.device)
        beta_prod_t = beta_prod_t[:,None,None,None].to(latents.device)
        denoised_latents = (noisy_latents - beta_prod_t ** (0.5) * noise_pred) / alpha_prod_t ** (0.5)
        
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
        self.controlnet.train()
        
    def set_eval(self):
        self.controlnet.eval()