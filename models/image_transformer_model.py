"""
Image-Transformer Diffusion Model.

Wraps the existing DiffusionModel (Transformer backbone) to operate on
image representations of time series instead of raw sequences.

Pipeline:
  ts (B, ch, seq_len)
    -> embedder.ts_to_img -> image (B, img_ch, H, W)
    -> flatten(2)         -> sequence (B, img_ch, H*W)
    -> DiffusionModel     -> sequence (B, img_ch, H*W)   [diffusion happens here]
    -> reshape            -> image (B, img_ch, H, W)
    -> embedder.img_to_ts -> ts (B, seq_len, ch)
    -> permute            -> ts (B, ch, seq_len)

Supports: "delay" (ch=6, H=W=8, seq=64) and "stft" (ch=12, H=8, W=9, seq=72).
"""

import torch
import torch.nn as nn
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.diffusion_model import DiffusionModel

def _get_embedder(embedder_type):
    if embedder_type == "delay":
        from utils.image_transforms import DelayEmbedder
        return DelayEmbedder
    elif embedder_type == "stft":
        try:
            from utils.image_transforms import STFTEmbedder
            return STFTEmbedder
        except ImportError as e:
            raise ImportError(f"STFT embedding requires torchaudio: pip install torchaudio. Error: {e}")


class ImageTransformerDiffusionModel(nn.Module):
    """
    Transformer diffusion model operating in flattened image space.
    The transformer is identical to raw mode — it just sees a different
    sequence length and channel count after the image embedding.
    """

    def __init__(self, config, embedder_type: str, device: str):
        super().__init__()
        self.embedder_type = embedder_type
        self.device = device

        seq_len = config.model.sequence_length   # raw ts length (32)
        channels = config.model.input_channels   # raw ts channels (6)

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
        else:
            raise ValueError(f"Unsupported embedder_type: {embedder_type}")

        # Probe image shape with a dummy forward pass (zero tensor)
        dummy_ts = torch.zeros(1, seq_len, channels, device=device)
        with torch.no_grad():
            dummy_img = self.embedder.ts_to_img(dummy_ts)

        _, _, H, W = dummy_img.shape
        self.H = H
        self.W = W
        self.img_channels = img_channels
        self.img_seq_len = H * W

        # Build transformer diffusion model with image-space dimensions
        self.diffusion_model = DiffusionModel(
            input_channels=img_channels,
            sequence_length=self.img_seq_len,
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
            device=device,
        )

    def compute_loss(self, batch_ts: torch.Tensor, loss_type: str = "l1") -> torch.Tensor:
        """
        Args:
            batch_ts: (B, channels, seq_len) from dataloader
        Returns:
            scalar loss
        """
        x_ts = batch_ts.permute(0, 2, 1).to(self.device)  # (B, seq_len, ch)
        x_img = self.embedder.ts_to_img(x_ts)              # (B, img_ch, H, W)
        x_flat = x_img.flatten(2)                          # (B, img_ch, H*W)
        return self.diffusion_model.compute_loss(x_flat, loss_type=loss_type)

    def sample(
        self,
        batch_size: int = 16,
        sampler_type: str = "ddim",
        num_steps: int = 50,
        eta: float = 0.0,
        return_trajectory: bool = False,
    ) -> torch.Tensor:
        """
        Generate synthetic time series.
        Returns: (batch_size, channels, seq_len)
        """
        flat = self.diffusion_model.sample(
            batch_size=batch_size,
            sampler_type=sampler_type,
            num_steps=num_steps,
            eta=eta,
            return_trajectory=return_trajectory,
        )  # (B, img_ch, H*W)

        if return_trajectory:
            return flat  # trajectory returned as-is (image-space)

        x_img = flat.reshape(batch_size, self.img_channels, self.H, self.W)
        x_ts = self.embedder.img_to_ts(x_img)  # (B, seq_len, ch)
        return x_ts.permute(0, 2, 1)           # (B, ch, seq_len)

    def to(self, device):
        super().to(device)
        self.device = device
        self.diffusion_model.to(device)
        self.embedder.device = device
        return self
