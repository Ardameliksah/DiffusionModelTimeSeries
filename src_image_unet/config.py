"""
Config for Image-UNet experiments.
Uses SongUNet backbone operating on image representations of time series.

Delay embedding:  ts(B,6,32) -> image(B,6,8,8)   -> UNet (img_resolution=8)
STFT embedding:   ts(B,6,32) -> image(B,12,8,9)  -> pad to (B,12,8,16) -> UNet (img_resolution=16)
"""

from dataclasses import dataclass
from typing import Literal, Tuple, Optional


@dataclass
class ImageEmbedConfig:
    """Image embedding hyperparameters (same as src_image)."""
    embedding_type: Literal["delay", "stft"] = "delay"
    # Delay params (seq_len=32 -> 8x8, 6 channels)
    delay: int = 4
    embedding_dim: int = 8
    # STFT params (seq_len=32 -> 8x9 -> padded to 8x16, 12 channels)
    n_fft: int = 14
    hop_length: int = 4


@dataclass
class UNetModelConfig:
    """SongUNet architecture. model_channels/num_blocks overridable via CLI."""
    model_channels: int = 64           # base channel width
    channel_mult: Tuple[int, ...] = (1, 2, 2, 2)  # per-level channel multipliers
    num_blocks: int = 2                # residual blocks per resolution level
    attn_resolutions: Optional[Tuple[int, ...]] = None  # None = auto (finest resolution)
    dropout: float = 0.1
    input_channels: int = 6            # raw ts channels
    sequence_length: int = 32          # raw ts length


@dataclass
class DiffusionConfig:
    num_timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    noise_schedule: Literal["linear", "cosine", "exponential"] = "cosine"
    variance_type: str = "fixed_large"
    gamma: float = 1.0


@dataclass
class TrainingConfig:
    batch_size: int = 64
    learning_rate: float = 5e-4
    num_epochs: int = 500
    warmup_steps: int = 100
    weight_decay: float = 1e-4
    gradient_clip_val: float = 1.0
    lr_scheduler_type: Literal["cosine", "linear"] = "cosine"
    checkpoint_dir: str = "MyCode/output/checkpoints_unet_delay"
    log_dir: str = "MyCode/output/logs_unet"
    save_every_n_epochs: int = 250
    validate_every_n_epochs: int = 10


@dataclass
class SamplingConfig:
    sampler_type: Literal["ddpm", "ddim"] = "ddim"
    num_sampling_steps: int = 100
    eta: float = 0.0
    batch_size: int = 16
    output_dir: str = "MyCode/output/generated_samples_unet"


@dataclass
class DataConfig:
    data_path: str = "MyCode/dataset/stocks_data.csv"
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    num_workers: int = 0
    pin_memory: bool = True
    neg_one_to_one: bool = True


class ImageUNetConfig:
    """Combined config for image-UNet pipeline."""

    def __init__(self):
        self.image = ImageEmbedConfig()
        self.model = UNetModelConfig()
        self.diffusion = DiffusionConfig()
        self.training = TrainingConfig()
        self.sampling = SamplingConfig()
        self.data = DataConfig()

    def to_dict(self) -> dict:
        return {
            "image": self.image.__dict__,
            "model": {k: list(v) if isinstance(v, tuple) else v
                      for k, v in self.model.__dict__.items()},
            "diffusion": self.diffusion.__dict__,
            "training": self.training.__dict__,
            "sampling": self.sampling.__dict__,
            "data": self.data.__dict__,
        }
