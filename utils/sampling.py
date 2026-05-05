"""
Sampling utilities for diffusion model generation.
Provides DDPM and DDIM samplers.
"""

import torch
from typing import Optional, Tuple
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from models import DiffusionModel


class DDPMSampler:
    """
    DDPM (Denoising Diffusion Probabilistic Models) sampler.
    Full reverse process - slower but stable.
    """
    
    def __init__(self, model: DiffusionModel):
        self.model = model
        self.diffusion = model.diffusion
    
    def sample(
        self,
        batch_size: int = 16,
        return_trajectory: bool = False,
    ) -> torch.Tensor:
        """
        Generate samples via full DDPM reverse process.
        
        Args:
            batch_size: Number of samples to generate
            return_trajectory: If True, return all T intermediate samples
        
        Returns:
            Tensor of shape (batch_size, channels, seq_len)
            or (T, batch_size, channels, seq_len) if return_trajectory=True
        """
        device = next(self.model.parameters()).device
        
        # Start with random noise
        x_t = torch.randn(
            batch_size,
            self.model.transformer.input_channels,
            self.model.transformer.sequence_length,
            device=device,
        )
        
        trajectory = [x_t.clone().detach().cpu()] if return_trajectory else None
        
        # Reverse diffusion: iterate from t=T-1 to t=0
        for t_idx in range(self.diffusion.num_timesteps - 1, -1, -1):
            t = torch.full((batch_size,), t_idx, dtype=torch.long, device=device)
            
            with torch.no_grad():
                # Predict noise then derive x_0
                noise_pred = self.model.transformer(x_t, t)
                x_start = self.diffusion.predict_start_from_noise(x_t, t, noise_pred)
                x_start.clamp_(-1.0, 1.0)

                # Compute posterior mean and log-variance
                mean, _, log_variance, _ = self.diffusion.p_mean_variance(
                    x_start, x_t, t, clip_denoised=False
                )

                # No noise at t=0
                nonzero_mask = (t > 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))

                # x_{t-1} = posterior_mean + sqrt(posterior_variance) * z
                z = torch.randn_like(x_t)
                x_t = mean + (0.5 * log_variance).exp() * nonzero_mask * z
            
            if return_trajectory:
                trajectory.append(x_t.clone().detach().cpu())
        
        if return_trajectory:
            return torch.stack(trajectory)
        return x_t


class DDIMSampler:
    """
    DDIM (Denoising Diffusion Implicit Models) sampler.
    Faster generation via timestep skipping - ~10-50x speedup.
    
    Implementation adapted from Diffusion-TS (https://github.com/Y-debug-sys/Diffusion-TS).
    Changes made: Added clamping for numerical stability (min=1e-8) to prevent division by small values.
    """
    
    def __init__(self, model: DiffusionModel):
        self.model = model
        self.diffusion = model.diffusion
    
    def sample(
        self,
        batch_size: int = 16,
        num_steps: int = 50,
        eta: float = 0.0,  # 0 = deterministic, 1 = stochastic (DDPM-like)
        return_trajectory: bool = False,
    ) -> torch.Tensor:
        """
        Generate samples via DDIM (faster, numerically stable).
        
        Args:
            batch_size: Number of samples to generate
            num_steps: Number of sampling steps (50-100 typical, << num_timesteps)
            eta: Stochasticity parameter (0 = deterministic, 1 = full stochasticity)
            return_trajectory: If True, return all intermediate samples
        
        Returns:
            Tensor of shape (batch_size, channels, seq_len)
            or (num_steps, batch_size, channels, seq_len) if return_trajectory=True
        """
        device = next(self.model.parameters()).device
        total_timesteps = self.diffusion.num_timesteps
        
        # Start with random noise
        x_t = torch.randn(
            batch_size,
            self.model.transformer.input_channels,
            self.model.transformer.sequence_length,
            device=device,
        )
        
        trajectory = [x_t.clone().detach().cpu()] if return_trajectory else None
        
        # Create timestep schedule: [-1, 0, 1, 2, ..., T-1] when num_steps == total_timesteps
        times = torch.linspace(-1, total_timesteps - 1, steps=num_steps + 1, device=device)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
        
        # DDIM sampling loop
        for time, time_next in time_pairs:
            time_cond = torch.full((batch_size,), time, dtype=torch.long, device=device)
            
            with torch.no_grad():
                # Predict noise using model
                noise_pred = self.model.transformer(x_t, time_cond)
                
                # Predict x_0 from noise prediction
                sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / torch.clamp(self.diffusion.alphas_cumprod[time], min=1e-8))
                sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / torch.clamp(self.diffusion.alphas_cumprod[time], min=1e-8) - 1)
                x_0_pred = sqrt_recip_alphas_cumprod * x_t - sqrt_recipm1_alphas_cumprod * noise_pred
                
                if time_next < 0:
                    x_t = x_0_pred
                    if return_trajectory:
                        trajectory.append(x_t.clone().detach().cpu())
                    continue
                
                # Get alphas for current and next timesteps
                alpha = self.diffusion.alphas_cumprod[time]
                alpha_next = self.diffusion.alphas_cumprod[time_next]
                
                # Compute sigma (stochasticity)
                sigma = eta * torch.sqrt((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha))
                
                # Compute c coefficient (numerically stable)
                c = torch.sqrt(1 - alpha_next - sigma ** 2)
                
                # DDIM update: x_{t-1} = sqrt(α̅_{t-1}) * x_0 + c * ε_θ + σ * z
                noise = torch.randn_like(x_t)
                x_t = torch.sqrt(alpha_next) * x_0_pred + c * noise_pred + sigma * noise
            
            if return_trajectory:
                trajectory.append(x_t.clone().detach().cpu())
        
        if return_trajectory:
            return torch.stack(trajectory)
        return x_t


def create_sampler(sampler_type: str = "ddim") -> type:
    """
    Factory function to create sampler class.
    
    Args:
        sampler_type: "ddpm" or "ddim"
    
    Returns:
        Sampler class
    """
    if sampler_type == "ddpm":
        return DDPMSampler
    elif sampler_type == "ddim":
        return DDIMSampler
    else:
        raise ValueError(f"Unknown sampler type: {sampler_type}")


def sample_batch(
    model: DiffusionModel,
    batch_size: int = 16,
    sampler_type: str = "ddim",
    num_steps: int = 50,
    eta: float = 0.0,
) -> torch.Tensor:
    """
    Convenience function to generate a batch of samples.
    
    Args:
        model: DiffusionModel instance
        batch_size: Number of samples
        sampler_type: "ddpm" or "ddim"
        num_steps: Number of sampling steps (for DDIM)
        eta: Stochasticity (for DDIM)
    
    Returns:
        Generated samples of shape (batch_size, channels, seq_len)
    """
    SamplerClass = create_sampler(sampler_type)
    sampler = SamplerClass(model)
    
    if sampler_type == "ddim":
        return sampler.sample(batch_size, num_steps=num_steps, eta=eta)
    else:
        return sampler.sample(batch_size)
