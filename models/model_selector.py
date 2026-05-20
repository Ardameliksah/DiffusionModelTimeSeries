"""
Unified model selector for RAW (Transformer) vs IMAGE (U-Net) mode.

- RAW mode: Direct transformer on time series (batch, channels, seq_len)
- IMAGE mode: U-Net on images with embeddings (batch, channels, H, W)

Both modes use the same training/sampling interface.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal, Tuple, Optional
from pathlib import Path

from models.diffusion_model import DiffusionModel
from models.unet_diffusion_model import UNetDiffusionModel
from utils.image_transforms import DelayEmbedder, PatchEmbedder, STFTEmbedder, MRTIEmbedder


def _next_power_of_2(n: int) -> int:
    """Return the smallest power of 2 that is >= n."""
    p = 1
    while p < n:
        p <<= 1
    return p


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
            from models.diffusion import GaussianDiffusion

            embedder_type = config.image.embedding_type
            input_channels = config.model.input_channels
            if embedder_type == "stft":
                input_channels = config.model.input_channels * 2

            # ── Build embedder FIRST so we can probe its actual output shape ──
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

            # Probe actual H×W produced by the embedder (e.g. STFT gives 8×9, not 8×8)
            _dummy = torch.zeros(1, config.model.sequence_length, config.model.input_channels)
            with torch.no_grad():
                _dummy_img = self.image_preprocessor.ts_to_img(_dummy)
            _, _, _H, _W = _dummy_img.shape
            self.orig_H = _H
            self.orig_W = _W
            # Pad each dim to the next power of 2 so UNet stride-2 ops are symmetric
            self.H_pad = _next_power_of_2(_H)
            self.W_pad = _next_power_of_2(_W)
            unet_res   = max(self.H_pad, self.W_pad)
            print(f"   [image mode] embedder output: {_H}×{_W}  →  padded to {self.H_pad}×{self.W_pad}  (UNet res={unet_res})")

            self.diffusion_model = UNetDiffusionModel(
                img_resolution=unet_res,
                in_channels=input_channels,
                out_channels=input_channels,
                model_channels=config.model.hidden_dim,
                num_blocks=config.model.num_layers,
                attn_resolutions=(unet_res,),
                dropout=config.model.dropout,
                embedding_type='fourier',
            )

            # ── Same GaussianDiffusion used by raw mode ───────────────────
            self.diffusion = GaussianDiffusion(
                num_timesteps=config.diffusion.num_timesteps,
                beta_start=config.diffusion.beta_start,
                beta_end=config.diffusion.beta_end,
                noise_schedule=config.diffusion.noise_schedule,
                device=device,
            )
            # Training objective & loss — read from config so they're saved
            # in the checkpoint and auto-restored at sampling/eval time.
            self.pred_objective = getattr(config.model, 'pred_objective', 'pred_eps')
            self.img_loss_type  = getattr(config.model, 'loss_type', 'mse')
            # ─────────────────────────────────────────────────────────────
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
            # Image mode: convert → pad → denoise → crop → convert back
            x_ts = x_t.permute(0, 2, 1)  # (batch, seq_len, channels)
            x_img = self.image_preprocessor.ts_to_img(x_ts)  # (batch, channels, H, W)
            x_img = F.pad(x_img, (0, self.W_pad - self.orig_W,
                                   0, self.H_pad - self.orig_H))  # (batch, C, H_pad, W_pad)
            output_img = self.diffusion_model(x_img, t)             # (batch, C, H_pad, W_pad)
            output_img = output_img[:, :, :self.orig_H, :self.orig_W]  # crop back
            output_ts = self.image_preprocessor.img_to_ts(output_img)  # (batch, seq_len, channels)
            return output_ts.permute(0, 2, 1)  # (batch, channels, seq_len)
        
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
            # ── Convert to image space ────────────────────────────────────
            x_ts  = batch.permute(0, 2, 1)                          # (B, L, C)
            x_0   = self.image_preprocessor.ts_to_img(x_ts)        # (B, C, H, W)  e.g. 8×9 for STFT

            # Pad to power-of-2 dims so UNet skip connections match on up/down pass
            x_0   = F.pad(x_0, (0, self.W_pad - self.orig_W,
                                  0, self.H_pad - self.orig_H))    # (B, C, H_pad, W_pad)

            B      = x_0.shape[0]
            device = x_0.device
            t      = torch.randint(0, self.diffusion.num_timesteps, (B,), device=device)
            x_t, noise = self.diffusion.q_sample(x_0, t)

            pred   = self.diffusion_model(x_t, t)                   # (B, C, H_pad, W_pad)

            # Target depends on prediction objective
            target = x_0 if self.pred_objective == "pred_x0" else noise

            # Base loss (use per-element so we can apply per-timestep weight)
            _fn    = F.l1_loss if self.img_loss_type == "l1" else F.mse_loss
            per_el = _fn(pred, target, reduction="none")            # (B, C, H_pad, W_pad)
            per_s  = per_el.mean(dim=list(range(1, per_el.ndim)))  # (B,)

            # Same per-timestep loss weighting as raw mode
            w = self.diffusion.loss_weight[t]                       # (B,)
            return (per_s * w).mean()
        
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

        # IMAGE mode — DDIM using the same GaussianDiffusion as raw mode
        channels = self.diffusion_model.in_channels
        total_T  = self.diffusion.num_timesteps

        # Sample in the padded space (H_pad×W_pad), which the UNet was built for
        x_t = torch.randn(batch_size, channels, self.H_pad, self.W_pad, device=device)
        trajectory = [x_t.cpu()] if return_trajectory else None

        # Build uniformly-spaced DDIM time pairs (same as DiffusionModel._sample_ddim)
        times      = torch.linspace(-1, total_T - 1, steps=num_steps + 1)
        times      = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        for time, time_next in time_pairs:
            t = torch.full((batch_size,), time, device=device, dtype=torch.long)

            with torch.no_grad():
                pred = self.diffusion_model(x_t, t)     # UNet output

            # Recover x_start depending on training objective
            if self.pred_objective == "pred_x0":
                x_start = pred.clamp(-1.0, 1.0)
            else:  # pred_eps
                x_start = self.diffusion.predict_start_from_noise(x_t, t, pred)
                x_start.clamp_(-1.0, 1.0)

            if time_next < 0:           # last step: output the denoised image directly
                x_t = x_start
                continue

            alpha      = self.diffusion.alphas_cumprod[time]
            alpha_next = self.diffusion.alphas_cumprod[time_next]
            sigma      = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c          = (1 - alpha_next - sigma ** 2).sqrt()

            noise_pred = self.diffusion.predict_noise_from_start(x_t, t, x_start)
            noise      = torch.randn_like(x_t)
            x_t        = x_start * alpha_next.sqrt() + c * noise_pred + sigma * noise

            if return_trajectory:
                trajectory.append(x_t.cpu())

        # Crop padding back to original embedder output shape before inverting
        x_t    = x_t[:, :, :self.orig_H, :self.orig_W]     # (batch, C, H, W)
        x_ts   = self.image_preprocessor.img_to_ts(x_t)    # (batch, seq_len, channels)
        output = x_ts.permute(0, 2, 1)                      # (batch, channels, seq_len)

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
        if hasattr(self, 'diffusion'):
            self.diffusion.to(device)
            self.diffusion.device = device
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
