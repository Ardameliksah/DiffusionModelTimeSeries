"""
Wrapper combining Transformer model with Diffusion process.
"""

import torch
import torch.nn as nn
from .diffusion import GaussianDiffusion
from .transformer import TransformerDiffusionModel


class DiffusionModel(nn.Module):
    """
    Complete diffusion model = Transformer + Gaussian Diffusion.
    Handles forward process (training) and provides interface for reverse process (sampling).
    """
    
    def __init__(
        self,
        # Transformer config
        input_channels: int = 6,
        sequence_length: int = 32,
        hidden_dim: int = 128,
        num_layers: int = 6,
        num_heads: int = 8,
        ff_dim: int = 512,
        dropout: float = 0.1,
        learnable_pos_enc: bool = True,
        # Diffusion config
        num_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        noise_schedule: str = "linear",
        device: str = "cpu",
    ):
        """
        Args:
            input_channels: Number of input channels (6)
            sequence_length: Time series length (32)
            hidden_dim: Transformer hidden dimension
            num_layers: Number of transformer blocks
            num_heads: Number of attention heads
            ff_dim: Feedforward dimension
            dropout: Dropout rate
            num_timesteps: Number of diffusion timesteps
            beta_start: Beta schedule start
            beta_end: Beta schedule end
            noise_schedule: "linear", "cosine", or "exponential"
            device: "cpu" or "cuda"
        """
        super().__init__()
        self.device = device
        
        # Initialize transformer model
        self.transformer = TransformerDiffusionModel(
            input_channels=input_channels,
            sequence_length=sequence_length,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout,
            learnable_pos_enc=learnable_pos_enc,
        )
        
        # Initialize diffusion process
        self.diffusion = GaussianDiffusion(
            num_timesteps=num_timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            noise_schedule=noise_schedule,
            device=device,
        )
    
    def forward(self, x_0: torch.Tensor, t: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass for training: predict noise from noisy sample.
        
        Args:
            x_0: Clean sample of shape (batch_size, channels, sequence_length)
            t: Optional timestep indices; if None, sample randomly
        
        Returns:
            Predicted noise of shape (batch_size, channels, sequence_length)
        """
        batch_size = x_0.shape[0]
        device = x_0.device
        
        # Sample random timesteps if not provided
        if t is None:
            t = torch.randint(0, self.diffusion.num_timesteps, (batch_size,), device=device)
        
        # Forward diffusion: add noise to x_0
        x_t, noise = self.diffusion.q_sample(x_0, t)
        
        # Predict noise using transformer
        predicted_noise = self.transformer(x_t, t)
        
        return predicted_noise, noise
    
    def compute_loss(self, x_0: torch.Tensor, loss_type: str = "mse") -> torch.Tensor:
        """
        Compute training loss with per-timestep reweighting (matches Diffusion-TS).
        
        Args:
            x_0: Clean sample
            loss_type: "mse" or "l1"
        
        Returns:
            Scalar loss value
        """
        batch_size = x_0.shape[0]
        device = x_0.device
        
        # Sample random timesteps
        t = torch.randint(0, self.diffusion.num_timesteps, (batch_size,), device=device)
        
        # Get predictions and target noise
        predicted_noise, target_noise = self.forward(x_0, t)
        
        # Compute per-element loss (no reduction yet)
        if loss_type == "mse":
            loss = nn.functional.mse_loss(predicted_noise, target_noise, reduction='none')
        elif loss_type == "l1":
            loss = nn.functional.l1_loss(predicted_noise, target_noise, reduction='none')
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
        
        # Reduce to per-sample loss: (B, C, L) -> (B,)
        loss = loss.mean(dim=list(range(1, loss.ndim)))
        
        # Per-timestep reweighting (matches Diffusion-TS loss_weight)
        loss = loss * self.diffusion.extract(self.diffusion.loss_weight, t, loss.shape).squeeze()
        
        return loss.mean()
    
    def sample(
        self,
        batch_size: int = 16,
        sampler_type: str = "ddim",
        num_steps: int = 50,
        eta: float = 0.0,
        return_trajectory: bool = False,
    ) -> torch.Tensor:
        """
        Generate samples via reverse diffusion process.
        
        Args:
            batch_size: Number of samples to generate
            sampler_type: "ddpm" (slow, stable) or "ddim" (fast)
            num_steps: Number of sampling steps (for DDIM)
            eta: Stochasticity parameter (0 = deterministic DDIM)
            return_trajectory: If True, return all intermediate samples
        
        Returns:
            Generated samples of shape (batch_size, channels, sequence_length)
            or (num_steps, batch_size, channels, sequence_length) if return_trajectory=True
        """
        device = next(self.transformer.parameters()).device
        
        # Start with random noise
        x_t = torch.randn(
            batch_size,
            self.transformer.input_channels,
            self.transformer.sequence_length,
            device=device,
        )
        
        if sampler_type == "ddpm":
            return self._sample_ddpm(x_t, return_trajectory)
        elif sampler_type == "ddim":
            return self._sample_ddim(x_t, num_steps=num_steps, eta=eta, return_trajectory=return_trajectory)
        else:
            raise ValueError(f"Unknown sampler type: {sampler_type}")
    
    def _predict_x_start(self, x_t: torch.Tensor, t: torch.Tensor, clip: bool = True):
        """Predict x_0 and noise from x_t. Matches Diffusion-TS model_predictions."""
        with torch.no_grad():
            noise_pred = self.transformer(x_t, t)
        x_start = self.diffusion.predict_start_from_noise(x_t, t, noise_pred)
        if clip:
            x_start.clamp_(-1.0, 1.0)
            # Re-derive noise from clipped x_start for consistency
            noise_pred = self.diffusion.predict_noise_from_start(x_t, t, x_start)
        return noise_pred, x_start

    def _sample_ddpm(self, x_t: torch.Tensor, return_trajectory: bool = False) -> torch.Tensor:
        """
        Full DDPM reverse process matching Diffusion-TS p_sample.
        Iterates from t=T-1 down to t=0.
        """
        trajectory = [x_t.cpu()] if return_trajectory else None
        device = x_t.device
        
        for t_idx in range(self.diffusion.num_timesteps - 1, -1, -1):
            t = torch.full((x_t.shape[0],), t_idx, dtype=torch.long, device=device)
            
            # Predict x_0 from current x_t
            _, x_start = self._predict_x_start(x_t, t, clip=True)
            
            # Compute posterior mean and variance
            model_mean, _, model_log_variance, _ = \
                self.diffusion.p_mean_variance(x_start, x_t, t, clip_denoised=True)
            
            # No noise at t=0
            noise = torch.randn_like(x_t) if t_idx > 0 else 0.0
            
            # x_{t-1} = posterior_mean + sqrt(posterior_variance) * z
            x_t = model_mean + (0.5 * model_log_variance).exp() * noise
            
            if return_trajectory:
                trajectory.append(x_t.cpu())
        
        if return_trajectory:
            return torch.stack(trajectory)
        return x_t
    
    def _sample_ddim(
        self,
        x_t: torch.Tensor,
        num_steps: int = 50,
        eta: float = 0.0,
        return_trajectory: bool = False,
    ) -> torch.Tensor:
        """
        DDIM reverse process matching Diffusion-TS fast_sample exactly.

        x_{t-1} = sqrt(alpha_prev) * x_0_pred
                 + sqrt(1 - alpha_prev - sigma^2) * pred_noise
                 + sigma * z
        """
        device = x_t.device
        batch = x_t.shape[0]
        trajectory = [x_t.cpu()] if return_trajectory else None
        total_timesteps = self.diffusion.num_timesteps

        # Build time pairs exactly like Diffusion-TS:
        # [-1, 0, 1, 2, ..., T-1] when num_steps == total_timesteps
        times = torch.linspace(-1, total_timesteps - 1, steps=num_steps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        for time, time_next in time_pairs:
            t = torch.full((batch,), time, device=device, dtype=torch.long)
            pred_noise, x_start = self._predict_x_start(x_t, t, clip=True)

            if time_next < 0:
                x_t = x_start
                continue

            alpha = self.diffusion.alphas_cumprod[time]
            alpha_next = self.diffusion.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(x_t)
            x_t = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise

            if return_trajectory:
                trajectory.append(x_t.cpu())

        if return_trajectory:
            return torch.stack(trajectory)
        return x_t
    
    def get_config(self) -> dict:
        """Return model configuration."""
        return {
            "transformer": {
                "input_channels": self.transformer.input_channels,
                "sequence_length": self.transformer.sequence_length,
                "hidden_dim": self.transformer.hidden_dim,
            },
            "diffusion": self.diffusion.get_noise_schedule_info(),
        }
