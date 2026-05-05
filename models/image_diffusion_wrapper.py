"""
Image-UNet Diffusion Model.

Wraps SongUNet to operate on 2D image representations of time series.

Pipeline:
  ts (B, ch, seq_len)
    -> embedder.ts_to_img -> image (B, img_ch, H, W)
    -> pad to power-of-2  -> (B, img_ch, H_pad, W_pad)
    -> SongUNet           -> predicted noise (B, img_ch, H_pad, W_pad)
    -> crop               -> (B, img_ch, H, W)
    -> embedder.img_to_ts -> ts (B, seq_len, ch)
    -> permute            -> ts (B, ch, seq_len)

Supports: "delay" (ch=6, H=W=8 -> 8x8, no pad) and "stft" (ch=12, H=8, W=9 -> pad to 8x16).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.unet_diffusion_model import SongUNet
from models.diffusion import GaussianDiffusion


def _get_embedder(embedder_type):
    if embedder_type == "delay":
        from utils.image_transforms import DelayEmbedder
        return DelayEmbedder
    elif embedder_type == "stft":
        try:
            from utils.image_transforms import STFTEmbedder
            return STFTEmbedder
        except ImportError as e:
            raise ImportError(f"STFT requires torchaudio: pip install torchaudio. Error: {e}")
    raise ValueError(f"Unknown embedder_type: {embedder_type}")


def _next_power_of_2(n):
    p = 1
    while p < n:
        p <<= 1
    return p


class ImageUNetDiffusionModel(nn.Module):
    """
    UNet diffusion model operating in 2D image space.
    The UNet backbone is identical to the original SongUNet — it just receives
    image representations of time series instead of natural images.
    """

    def __init__(self, config, embedder_type: str, device: str):
        super().__init__()
        self.embedder_type = embedder_type
        self.device = device

        seq_len = config.model.sequence_length
        channels = config.model.input_channels

        # Build embedder
        EmbedderClass = _get_embedder(embedder_type)
        if embedder_type == "delay":
            self.embedder = EmbedderClass(
                device=device,
                seq_len=seq_len,
                delay=config.image.delay,
                embedding=config.image.embedding_dim,
            )
            img_channels = channels
        elif embedder_type == "stft":
            self.embedder = EmbedderClass(
                device=device,
                seq_len=seq_len,
                n_fft=config.image.n_fft,
                hop_length=config.image.hop_length,
            )
            img_channels = channels * 2  # real + imaginary

        # Probe image shape with a dummy forward pass
        dummy_ts = torch.zeros(1, seq_len, channels, device=device)
        with torch.no_grad():
            dummy_img = self.embedder.ts_to_img(dummy_ts)
        _, _, H, W = dummy_img.shape
        self.H = H
        self.W = W
        self.img_channels = img_channels

        # Pad H, W to next power of 2 so UNet downsampling is symmetric
        H_pad = _next_power_of_2(H)
        W_pad = _next_power_of_2(W)
        self.H_pad = H_pad
        self.W_pad = W_pad
        unet_res = max(H_pad, W_pad)
        self.unet_res = unet_res

        # Use config attn_resolutions or default to finest resolution
        attn_res = getattr(config.model, "attn_resolutions", None)
        if attn_res is None:
            attn_res = [unet_res]

        self.unet = SongUNet(
            img_resolution=unet_res,
            in_channels=img_channels,
            out_channels=img_channels,
            model_channels=config.model.model_channels,
            channel_mult=list(config.model.channel_mult),
            num_blocks=config.model.num_blocks,
            attn_resolutions=list(attn_res),
            dropout=config.model.dropout,
        )

        self.diffusion = GaussianDiffusion(
            num_timesteps=config.diffusion.num_timesteps,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            noise_schedule=config.diffusion.noise_schedule,
            device=device,
        )

    def _pad(self, x: torch.Tensor) -> torch.Tensor:
        return F.pad(x, (0, self.W_pad - self.W, 0, self.H_pad - self.H))

    def _crop(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, : self.H, : self.W]

    def compute_loss(self, batch_ts: torch.Tensor, loss_type: str = "mse") -> torch.Tensor:
        """
        Args:
            batch_ts: (B, channels, seq_len) from dataloader
        Returns:
            scalar loss
        """
        x_0_ts = batch_ts.permute(0, 2, 1).to(self.device)
        x_0_img = self.embedder.ts_to_img(x_0_ts)   # (B, img_ch, H, W)
        x_0_img = self._pad(x_0_img)                 # (B, img_ch, H_pad, W_pad)

        B = x_0_img.shape[0]
        t = torch.randint(0, self.diffusion.num_timesteps, (B,), device=self.device)
        x_t, noise = self.diffusion.q_sample(x_0_img, t)

        # SongUNet forward: (x, noise_labels, class_labels)
        # noise_labels: float timestep values used by positional embedding
        pred_noise = self.unet(x_t, t.float(), class_labels=None)

        if loss_type == "mse":
            loss = F.mse_loss(pred_noise, noise, reduction="none")
        elif loss_type == "l1":
            loss = F.l1_loss(pred_noise, noise, reduction="none")
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

        # Per-sample mean, then per-timestep reweighting (matches DiffusionModel pattern)
        loss = loss.mean(dim=list(range(1, loss.ndim)))   # (B,)
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
        Generate synthetic time series via reverse diffusion.
        Returns: (batch_size, channels, seq_len)
        """
        x_t = torch.randn(
            batch_size, self.img_channels, self.H_pad, self.W_pad,
            device=self.device,
        )

        if sampler_type == "ddpm":
            result = self._sample_ddpm(x_t, return_trajectory)
        elif sampler_type == "ddim":
            result = self._sample_ddim(x_t, num_steps, eta, return_trajectory)
        else:
            raise ValueError(f"Unknown sampler: {sampler_type}")

        if return_trajectory:
            return result  # image-space trajectory

        x_img = self._crop(result)
        x_ts = self.embedder.img_to_ts(x_img)  # (B, seq_len, ch)
        return x_ts.permute(0, 2, 1)           # (B, ch, seq_len)

    def _predict_x_start(self, x_t, t, clip=True):
        with torch.no_grad():
            noise_pred = self.unet(x_t, t.float(), class_labels=None)
        x_start = self.diffusion.predict_start_from_noise(x_t, t, noise_pred)
        if clip:
            x_start.clamp_(-1.0, 1.0)
            noise_pred = self.diffusion.predict_noise_from_start(x_t, t, x_start)
        return noise_pred, x_start

    def _sample_ddpm(self, x_t, return_trajectory=False):
        trajectory = [x_t.cpu()] if return_trajectory else None
        device = x_t.device
        for t_idx in range(self.diffusion.num_timesteps - 1, -1, -1):
            t = torch.full((x_t.shape[0],), t_idx, dtype=torch.long, device=device)
            _, x_start = self._predict_x_start(x_t, t, clip=True)
            model_mean, _, model_log_var, _ = self.diffusion.p_mean_variance(
                x_start, x_t, t, clip_denoised=True
            )
            noise = torch.randn_like(x_t) if t_idx > 0 else 0.0
            x_t = model_mean + (0.5 * model_log_var).exp() * noise
            if return_trajectory:
                trajectory.append(x_t.cpu())
        return torch.stack(trajectory) if return_trajectory else x_t

    def _sample_ddim(self, x_t, num_steps=50, eta=0.0, return_trajectory=False):
        device = x_t.device
        batch = x_t.shape[0]
        trajectory = [x_t.cpu()] if return_trajectory else None
        total_timesteps = self.diffusion.num_timesteps

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

        return torch.stack(trajectory) if return_trajectory else x_t

    def to(self, device):
        super().to(device)
        self.device = device
        self.diffusion.device = device
        self.embedder.device = device
        return self
