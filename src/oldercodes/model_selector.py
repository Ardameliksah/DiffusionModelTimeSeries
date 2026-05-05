"""
Unified model selector for RAW (Transformer) vs IMAGE (U-Net) mode.

- RAW mode: Direct transformer on time series (batch, channels, seq_len)
- IMAGE mode: U-Net on images with embeddings (batch, channels, H, W)

Both modes use the same training/sampling interface.
"""

import torch
import torch.nn as nn
from typing import Literal, Tuple, Optional
from pathlib import Path

from models.diffusion_model import DiffusionModel
from models.unet_diffusion_model import UNetDiffusionModel
from utils.image_transforms import DelayEmbedder, PatchEmbedder, STFTEmbedder, MRTIEmbedder


class ImagePreprocessor(nn.Module):
    """Time series ↔ Image conversion wrapper."""
    
    def __init__(self, embedder_type: Literal["delay", "patch", "stft", "mrti"], device: str, **embedder_kwargs):
        super().__init__()
        self.embedder_type = embedder_type
        self.device = device
        
        if embedder_type == "delay":
            self.embedder = DelayEmbedder(
                device=device,
                seq_len=embedder_kwargs.get("seq_len", 32),
                delay=embedder_kwargs.get("delay", 4),
                embedding=embedder_kwargs.get("embedding_dim", 8)
            )
        elif embedder_type == "patch":
            self.embedder = PatchEmbedder(
                device=device,
                seq_len=embedder_kwargs.get("seq_len", 32),
                patch_size=embedder_kwargs.get("patch_size", 4),
                img_size=embedder_kwargs.get("img_size", 8)
            )
        elif embedder_type == "stft":
            self.embedder = STFTEmbedder(
                device=device,
                seq_len=embedder_kwargs.get("seq_len", 32),
                n_fft=embedder_kwargs.get("n_fft", 16),
                hop_length=embedder_kwargs.get("hop_length", 4)
            )
        elif embedder_type == "mrti":
            self.embedder = MRTIEmbedder(
                device=device,
                seq_len=embedder_kwargs.get("seq_len", 32),
                num_scales=embedder_kwargs.get("num_scales", 3),
                num_periods=embedder_kwargs.get("num_periods", 3)
            )
        else:
            raise ValueError(f"Unknown embedder type: {embedder_type}")
    
    def ts_to_img(self, x_ts):
        """Time series → Image. Input: (batch, seq_len, channels) → Output: (batch, channels, H, W)"""
        return self.embedder.ts_to_img(x_ts)
    
    def img_to_ts(self, x_img):
        """Image → Time series. Input: (batch, channels, H, W) → Output: (batch, seq_len, channels)"""
        return self.embedder.img_to_ts(x_img)
    
    def to(self, device):
        """Move to device and update internal device reference."""
        super().to(device)
        self.device = device
        # Ensure embedder uses same device
        if hasattr(self.embedder, 'device'):
            self.embedder.device = device
        return self


class UnifiedDiffusionModel(nn.Module):
    """
    Unified interface for both RAW and IMAGE mode models.
    
    Modes:
    - RAW: Uses transformer directly on time series
    - IMAGE: Uses U-Net on image representation with embeddings
    """
    
    def __init__(
        self,
        model_type: Literal["raw", "image"],
        config,
        device: str,
    ):
        super().__init__()
        self.model_type = model_type
        self.config = config
        self.device = device
        
        if model_type == "raw":
            # Transformer mode
            self.diffusion_model = DiffusionModel(
                input_channels=config.model.input_channels,
                sequence_length=config.model.sequence_length,
                hidden_dim=config.model.hidden_dim,
                num_layers=config.model.num_layers,
                num_heads=config.model.num_heads,
                ff_dim=config.model.ff_dim,
                dropout=config.model.dropout,
                learnable_pos_enc=config.model.learnable_pos_enc,
                num_timesteps=config.diffusion.num_timesteps,
                beta_start=config.diffusion.beta_start,
                beta_end=config.diffusion.beta_end,
                noise_schedule=config.diffusion.noise_schedule,
            )
            self.image_preprocessor = None
        
        elif model_type == "image":
            # U-Net mode with image embeddings
            # Determine input channels based on embedder type
            embedder_type = config.image.embedding_type
            input_channels = config.model.input_channels
            
            # STFT doubles channels (real + imaginary concatenated)
            if embedder_type == "stft":
                input_channels = config.model.input_channels * 2
            
            self.diffusion_model = UNetDiffusionModel(
                in_channels=input_channels,
                out_channels=input_channels,
                model_channels=config.model.hidden_dim,
                num_blocks=config.model.num_layers,
                attn_resolutions=(8,),
                dropout=config.model.dropout,
                embedding_type='fourier',
            )
            
            # Image embedder
            self.image_preprocessor = ImagePreprocessor(
                embedder_type=embedder_type,
                device=device,
                seq_len=config.model.sequence_length,
                delay=getattr(config.image, "delay", 4),
                embedding_dim=getattr(config.image, "embedding_dim", 8),
                patch_size=getattr(config.image, "patch_size", 4),
                img_size=getattr(config.image, "img_size", 8),
                n_fft=getattr(config.image, "n_fft", 16),
                hop_length=getattr(config.image, "hop_length", 4),
                num_scales=getattr(config.image, "num_scales", 3),
                num_periods=getattr(config.image, "num_periods", 3),
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through diffusion model.
        
        RAW mode:
            Input: (batch, channels, seq_len) → Output: (batch, channels, seq_len)
        
        IMAGE mode:
            Input: (batch, channels, seq_len) → Image (batch, channels, H, W) → 
            Denoising → Image (batch, channels, H, W) → Time series (batch, channels, seq_len)
        """
        if self.model_type == "raw":
            # Direct transformer mode
            return self.diffusion_model(x_t, t)
        
        elif self.model_type == "image":
            # Image mode: convert → denoise → convert back
            batch_size = x_t.shape[0]
            
            # Convert time series to image
            x_ts = x_t.permute(0, 2, 1)  # (batch, seq_len, channels)
            x_img = self.image_preprocessor.ts_to_img(x_ts)  # (batch, channels, H, W)
            
            # Denoise through U-Net
            output_img = self.diffusion_model(x_img, t)  # (batch, channels, H, W)
            
            # Convert back to time series
            output_ts = self.image_preprocessor.img_to_ts(output_img)  # (batch, seq_len, channels)
            output = output_ts.permute(0, 2, 1)  # (batch, channels, seq_len)
            
            return output
        
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def compute_loss(self, batch: torch.Tensor, loss_type: str = "mse") -> torch.Tensor:
        """
        Compute diffusion loss.
        
        Both modes compute loss appropriately:
        - RAW mode: Loss on time series directly
        - IMAGE mode: Convert to image, compute loss on image space
        
        Args:
            batch: (batch, channels, seq_len) from dataloader
            loss_type: "mse" or "l1"
        
        Returns:
            Scalar loss
        """
        if self.model_type == "raw":
            # Transformer loss on time series
            return self.diffusion_model.compute_loss(batch, loss_type=loss_type)
        
        elif self.model_type == "image":
            # Convert to image space and compute loss there
            x_ts = batch.permute(0, 2, 1)  # (batch, seq_len, channels)
            x_img = self.image_preprocessor.ts_to_img(x_ts)  # (batch, channels, H, W)
            return self.diffusion_model.compute_loss(x_img, loss_type=loss_type)
        
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def sample(
        self,
        batch_size: int = 16,
        sampler_type: str = "ddim",
        num_steps: int = 50,
        eta: float = 0.0,
        return_trajectory: bool = False,
    ) -> torch.Tensor:
        """
        Generate synthetic time series samples.

        RAW mode: delegates to DiffusionModel.sample() (full DDPM/DDIM pipeline).
        IMAGE mode: runs reverse diffusion in image space using the linear-sigma
                    schedule from UNetDiffusionModel.compute_loss, then converts
                    back to time series via the image preprocessor.

        Returns:
            (batch_size, channels, seq_len)
        """
        device = next(self.parameters()).device

        if self.model_type == "raw":
            return self.diffusion_model.sample(
                batch_size=batch_size,
                sampler_type=sampler_type,
                num_steps=num_steps,
                eta=eta,
                return_trajectory=return_trajectory,
            )

        # IMAGE mode: reverse the linear-sigma forward process
        # Forward: x_t = x_0 + (t/T) * noise  →  reverse via Euler steps
        img_res = self.diffusion_model.img_resolution
        channels = self.diffusion_model.in_channels
        T = self.diffusion_model.num_timesteps

        x_t = torch.randn(batch_size, channels, img_res, img_res, device=device)
        trajectory = [x_t.cpu()] if return_trajectory else None

        step_indices = list(range(T - 1, -1, -max(1, T // num_steps)))
        for t_idx in step_indices:
            t = torch.full((batch_size,), t_idx, dtype=torch.long, device=device)
            with torch.no_grad():
                noise_pred = self.diffusion_model(x_t, t)

            sigma_t = t_idx / T
            sigma_prev = max(t_idx - max(1, T // num_steps), 0) / T
            x_t = x_t - (sigma_t - sigma_prev) * noise_pred

            if return_trajectory:
                trajectory.append(x_t.cpu())

        # Convert generated image back to time series
        x_ts = self.image_preprocessor.img_to_ts(x_t)   # (batch, seq_len, channels)
        output = x_ts.permute(0, 2, 1)                   # (batch, channels, seq_len)

        if return_trajectory:
            return torch.stack(trajectory)
        return output

    def get_diffusion_model(self):
        """Get underlying diffusion model for optimizer access."""
        return self.diffusion_model
    
    def get_parameters(self):
        """Get all trainable parameters."""
        params = list(self.diffusion_model.parameters())
        if self.image_preprocessor is not None:
            params += list(self.image_preprocessor.embedder.parameters())
        return params
    
    def to(self, device):
        """Move model to device."""
        super().to(device)
        self.diffusion_model.to(device)
        if self.image_preprocessor is not None:
            self.image_preprocessor.to(device)
        return self


def create_model(
    config,
    model_type: Literal["raw", "image"],
    device: str
) -> UnifiedDiffusionModel:
    """
    Factory function to create unified model.
    
    Args:
        config: Configuration object (Config or ImageVersionConfig)
        model_type: "raw" for transformer, "image" for U-Net
        device: "cpu" or "cuda"
    
    Returns:
        UnifiedDiffusionModel ready for training/sampling
    """
    model = UnifiedDiffusionModel(
        model_type=model_type,
        config=config,
        device=device
    )
    return model.to(device)
