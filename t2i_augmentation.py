import torch
import torch.nn.functional as F

from diffusers import T2IAdapter 

from base_augmentation import BaseAugmentation, DenoiseType, SchedulerType

from tqdm import tqdm

class T2IAdapterAugmentation(BaseAugmentation):
    def __init__(self, 
                 model_path, 
                 scheduler_type=SchedulerType.DDIM,
                 eta=0.0,
                 max_timesteps=1000,
                 num_timesteps=20,
                 device="cuda"):
        super().__init__(scheduler_type, eta, max_timesteps, num_timesteps, device, DenoiseType.PARTIAL_DENOISE_T_FIXED)
        
        t2i_adapter = T2IAdapter.from_pretrained(model_path)
                
        self.set_condition_model(t2i_adapter)
        self.set_train()
        
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
                        
        # if condition has more than 3 channels, take only one
        cond_size = condition.size()
        if cond_size[1] > 1:
            condition = condition[:,0:1]
               
        # encode images into the latent space
        latents = self.vae.encode(image).latent_dist.sample()
        latents *= self.vae.config.scaling_factor
        
        # sample noise to be added to the latents
        noise = torch.randn_like(latents)
        
        # Sample a random timestep for each image
        batch_size = latents.shape[0]
        if self.is_eval:           

            #NOTE: Method 3
            # self.scheduler.set_timesteps(self.num_timesteps, device=latents.device)
            # timesteps = self.scheduler.timesteps
            # noisy_latents = self.scheduler.add_noise(latents, noise, timesteps[0].repeat(batch_size))
            
            #NOTE: Method 4. Initial timesteps changes the texture, we are after them.
            self.scheduler.config.num_train_timesteps = self.max_timesteps
            self.scheduler.set_timesteps(self.num_timesteps, device=latents.device)
            timesteps = self.scheduler.timesteps
            noisy_latents = self.scheduler.add_noise(latents, noise, timesteps[0].repeat(batch_size))
        else:
            #NOTE: Method 1
            #! look out, this method expects a different kind of treatment in the eval part
            timesteps = torch.tensor([self.max_timesteps] * batch_size, device=latents.device)
            
            #NOTE: Method 2
            #TODO investigate, should use num_inference_step or scheduler.config.num_train_timesteps
            # timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, (batch_size,), device=latents.device)
            
            #NOTE: Method 3
            # self.scheduler.set_timesteps(self.num_timesteps, device=latents.device)
            # indices = torch.randint(len(self.scheduler.timesteps), (batch_size,))
            # timesteps = self.scheduler.timesteps[indices]
            
            #NOTE: Method 4. Initial timesteps changes the texture, we are after them.
            # self.scheduler.config.num_train_timesteps = self.max_timesteps
            # self.scheduler.set_timesteps(self.num_timesteps, device=latents.device)
            # indices = torch.randint(len(self.scheduler.timesteps), (batch_size,))
            # timesteps = self.scheduler.timesteps[indices]            
            
            noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)
        
        # hacky way of combining train with validation
        if not self.is_eval:
            timesteps = timesteps[None]
            
        #NOTE [DEBUG]: the result of this opperation occupies ~14Gb of VRAM 
        adapter_state = self.condition_model(condition)
        adapter_state = [state.to(dtype=torch.float32) for state in adapter_state]        
            
        loss = 0.
        timesteps = tqdm(timesteps, leave=None, desc="denoising step-by-step") if self.is_eval else timesteps
        for t in timesteps:
            # Predict the noise residual
            #NOTE [DEBUG]: the result of this opperation occupies ~53Gb of VRAM 
            noise_pred = self.unet(
                noisy_latents,
                t,
                encoder_hidden_states=embeddings,
                down_block_additional_residuals=[state.clone() for state in adapter_state],
            ).sample
                    
            # perfom 1 denoise step
            if self.is_eval:
                if self.scheduler_type == SchedulerType.DDIM:
                    noisy_latents = self.scheduler.step(noise_pred, t, noisy_latents, eta=self.eta, return_dict=False)[0]
                elif self.scheduler_type == SchedulerType.DDPM:
                    noisy_latents = self.scheduler.step(noise_pred, t, noisy_latents, return_dict=False)[0]
            else:
                # Get the target for loss depending on the prediction type
                if self.scheduler.config.prediction_type == "epsilon":
                    target = noise
                else:
                    raise ValueError(f"Unknown prediction type {self.scheduler.config.prediction_type}")
                loss = F.mse_loss(noise_pred.float(), target.float(), reduction="mean")

                # perfom x0 rough prediction (xt ➙ x0 in 1 step)
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
        
        return diffusion_pred, loss