"""
Gaussian Diffusion Process for Time Series.
Implements forward process (adding noise), reverse process, and three noise schedules:
- Linear: standard linear interpolation
- Cosine: smoother signal preservation
- Exponential: aggressive early reduction
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Literal
import numpy as np


class GaussianDiffusion(nn.Module):
    """
    Gaussian diffusion process for time series synthesis.
    
    Forward process (q): adds Gaussian noise to clean samples over T timesteps
    Reverse process (p_θ): learns to remove noise iteratively
    """
    
    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        noise_schedule: Literal["linear", "cosine", "exponential"] = "linear",
        variance_type: Literal["fixed_large", "fixed_small", "learned_range"] = "fixed_large",
        gamma: float = 1.0,
        device: str = "cpu",
    ):
        """
        Args:
            num_timesteps: Number of diffusion timesteps (T), typically 1000
            beta_start: Starting variance schedule value
            beta_end: Ending variance schedule value
            noise_schedule: Type of noise schedule to use
            variance_type: How to handle variance in reverse process
            gamma: Exponential schedule parameter (only used if noise_schedule="exponential")
            device: Device to create tensors on (cpu/cuda)
        """
        super().__init__()
        
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.noise_schedule = noise_schedule
        self.variance_type = variance_type
        self.gamma = gamma
        self.device = device
        
        # Compute noise schedule and pre-compute diffusion coefficient buffers
        self._create_schedules()
    
    def _create_schedules(self):
        """Create noise schedule and pre-compute all required buffers."""
        
        if self.noise_schedule == "linear":
            # Match Diffusion-TS: scale by 1000/T so schedule is invariant to num_timesteps
            scale = 1000 / self.num_timesteps
            beta_start = scale * self.beta_start
            beta_end = scale * self.beta_end
            betas = torch.linspace(beta_start, beta_end, self.num_timesteps, dtype=torch.float64, device=self.device)
        
        elif self.noise_schedule == "cosine":
            # Cosine schedule from "Improved Denoising Diffusion Probabilistic Models"
            s = 0.008
            steps = torch.arange(self.num_timesteps + 1, device=self.device, dtype=torch.float64)
            alphas_cumprod = torch.cos(((steps / self.num_timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = torch.clip(betas, 0.0001, 0.9999)
        
        elif self.noise_schedule == "exponential":
            # Exponential schedule: β_t = β_start + (β_end - β_start) * (1 - exp(-γ*t/T))
            t = torch.arange(self.num_timesteps, device=self.device, dtype=torch.float32)
            normalized_t = t / self.num_timesteps
            betas = self.beta_start + (self.beta_end - self.beta_start) * (1 - torch.exp(-self.gamma * normalized_t))
        
        else:
            raise ValueError(f"Unknown noise schedule: {self.noise_schedule}")
        
        # Ensure betas are in valid range
        betas = torch.clip(betas, 1e-5, 0.9999).float()
        
        # Register as buffer (not learnable parameters)
        self.register_buffer("betas", betas)
        
        # Compute alphas (complement of betas)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1, device=self.device), alphas_cumprod[:-1]], dim=0)
        
        # Forward process coefficients
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        
        # Variance of forward process q(x_t | x_{t-1})
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        
        # Coefficients for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        self.register_buffer("posterior_variance", posterior_variance)
        
        # Clipped posterior variance (to avoid log(0))
        posterior_variance_clipped = torch.clamp(posterior_variance, min=1e-20)
        self.register_buffer("posterior_log_variance_clipped", torch.log(posterior_variance_clipped))
        
        # Coefficient for posterior mean
        self.register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )
        
        # For predicting x_0 from noise and vice versa
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1))
        
        # Per-timestep loss reweighting (matches Diffusion-TS)
        self.register_buffer(
            "loss_weight",
            torch.sqrt(alphas) * torch.sqrt(1.0 - alphas_cumprod) / betas / 100,
        )
    
    def extract(self, arr: torch.Tensor, timesteps: torch.Tensor, x_shape: Tuple) -> torch.Tensor:
        """
        Extract coefficients at specified timesteps, then reshape to broadcast with x.
        
        Args:
            arr: 1D tensor of shape (num_timesteps,) with coefficient values
            timesteps: Tensor of shape (batch_size,) with timestep indices
            x_shape: Shape of the input tensor x_t, e.g., (batch_size, channels, sequence_length)
        
        Returns:
            Tensor of shape (batch_size, 1, 1, ...) with expanded dimensions for broadcasting
        """
        batch_size = timesteps.shape[0]
        out = arr.gather(0, timesteps.to(arr.device))  # (batch_size,)
        
        # Reshape to (batch_size, 1, 1, ...) for broadcasting
        return out.reshape(batch_size, *([1] * (len(x_shape) - 1)))
    
    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward diffusion process: add noise to clean sample x_0 at timestep t.
        
        x_t = sqrt(α̅_t) * x_0 + sqrt(1 - α̅_t) * ε
        
        Args:
            x_0: Clean sample of shape (batch_size, channels, sequence_length)
            t: Timestep indices of shape (batch_size,)
            noise: Optional noise tensor; if None, sample from standard Gaussian
        
        Returns:
            Tuple of (x_t, noise) where x_t is the noisy sample at timestep t
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        
        # Extract coefficients for the given timesteps
        sqrt_alphas_t = self.extract(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_t = self.extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)
        
        # Compute noisy sample
        x_t = sqrt_alphas_t * x_0 + sqrt_one_minus_alphas_t * noise
        
        return x_t, noise
    
    def q_posterior_mean_variance(
        self, x_0: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute posterior distribution q(x_{t-1} | x_t, x_0).
        
        This is used during training to compute the target for the reverse process.
        
        Args:
            x_0: Clean sample
            x_t: Noisy sample at timestep t
            t: Timestep indices
        
        Returns:
            Tuple of (posterior_mean, posterior_variance)
        """
        posterior_mean = (
            self.extract(self.posterior_mean_coef1, t, x_t.shape) * x_0
            + self.extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = self.extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = self.extract(self.posterior_log_variance_clipped, t, x_t.shape)
        
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    
    def predict_start_from_noise(self, x_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """
        Predict x_0 from x_t and predicted noise.
        x_0 = (1/√α̅_t) * x_t  -  √(1/α̅_t - 1) * ε
        """
        return (
            self.extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - self.extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t: torch.Tensor, t: torch.Tensor, x_0: torch.Tensor) -> torch.Tensor:
        """
        Predict noise from x_t and predicted x_0.
        ε = ((1/√α̅_t) * x_t  -  x_0) / √(1/α̅_t - 1)
        """
        return (
            (self.extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x_0)
            / self.extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def p_mean_variance(
        self, x_start: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor, clip_denoised: bool = True
    ):
        """
        Compute mean and variance of reverse process p(x_{t-1} | x_t).

        Uses the correct posterior formula (matches Diffusion-TS / TransFusion / PaD-TS):
          1. Predict x_0 from noise
          2. Optionally clip x_0 to [-1, 1]
          3. Compute posterior mean = coef1 * x_0 + coef2 * x_t

        Args:
            x_start: Predicted x_0 (already computed by caller)
            x_t: Noisy sample at timestep t
            t: Timestep indices
            clip_denoised: Whether to clip predicted x_0 to [-1, 1]

        Returns:
            (model_mean, posterior_variance, posterior_log_variance_clipped, x_start)
        """
        if clip_denoised:
            x_start.clamp_(-1.0, 1.0)

        model_mean, posterior_variance, posterior_log_variance_clipped = \
            self.q_posterior_mean_variance(x_0=x_start, x_t=x_t, t=t)

        return model_mean, posterior_variance, posterior_log_variance_clipped, x_start
    
    def compute_loss(
        self,
        model: nn.Module,
        x_0: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor = None,
        loss_type: str = "mse",
    ) -> torch.Tensor:
        """
        Compute training loss for the diffusion model.
        
        Loss = ||ε - ε_θ(x_t, t)||^2 where x_t = sqrt(α̅_t)*x_0 + sqrt(1-α̅_t)*ε
        
        Args:
            model: Diffusion model (transformer) that predicts noise
            x_0: Clean sample
            t: Timestep indices
            noise: Optional noise (if None, will be sampled)
            loss_type: Type of loss ("mse", "l2", "smooth_l1")
        
        Returns:
            Scalar loss value (averaged over batch)
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        
        # Forward diffusion: add noise to x_0
        x_t, _ = self.q_sample(x_0, t, noise)
        
        # Predict noise using model
        predicted_noise = model(x_t, t)
        
        # Compute loss
        if loss_type == "mse":
            loss = torch.mean((noise - predicted_noise) ** 2)
        elif loss_type == "l1":
            loss = torch.mean(torch.abs(noise - predicted_noise))
        elif loss_type == "smooth_l1":
            loss = torch.nn.functional.smooth_l1_loss(predicted_noise, noise)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
        
        return loss
    
    def get_noise_schedule_info(self) -> Dict:
        """Return information about the current noise schedule."""
        return {
            "schedule_type": self.noise_schedule,
            "num_timesteps": self.num_timesteps,
            "beta_start": self.beta_start,
            "beta_end": self.beta_end,
            "max_beta": float(self.betas.max()),
            "min_beta": float(self.betas.min()),
            "max_alpha_cumprod": float(self.alphas_cumprod.max()),
            "min_alpha_cumprod": float(self.alphas_cumprod.min()),
        }


def create_diffusion(
    num_timesteps: int = 1000,
    beta_start: float = 1e-4,
    beta_end: float = 2e-2,
    noise_schedule: str = "linear",
    variance_type: str = "fixed_large",
    gamma: float = 1.0,
    device: str = "cpu",
) -> GaussianDiffusion:
    """Convenience function to create a GaussianDiffusion instance."""
    return GaussianDiffusion(
        num_timesteps=num_timesteps,
        beta_start=beta_start,
        beta_end=beta_end,
        noise_schedule=noise_schedule,
        variance_type=variance_type,
        gamma=gamma,
        device=device,
    )


if __name__ == "__main__":
    # Test diffusion process
    device = "cpu"
    diffusion = GaussianDiffusion(
        num_timesteps=100,
        beta_start=1e-4,
        beta_end=2e-2,
        noise_schedule="linear",
        device=device,
    )
    
    # Create dummy data
    batch_size, channels, seq_len = 16, 6, 32
    x_0 = torch.randn(batch_size, channels, seq_len)
    t = torch.randint(0, 100, (batch_size,))
    noise = torch.randn_like(x_0)
    
    # Test forward diffusion
    x_t, _ = diffusion.q_sample(x_0, t, noise)
    print(f"x_0 shape: {x_0.shape}, range: [{x_0.min():.4f}, {x_0.max():.4f}]")
    print(f"x_t shape: {x_t.shape}, range: [{x_t.min():.4f}, {x_t.max():.4f}]")
    
    # Test noise schedule
    schedule_info = diffusion.get_noise_schedule_info()
    print(f"\nNoise schedule info:")
    for k, v in schedule_info.items():
        print(f"  {k}: {v}")
    
    print("\nDiffusion test passed!")
