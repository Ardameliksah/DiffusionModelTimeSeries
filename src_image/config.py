"""
Config for Image-Transformer experiments.
Reuses the same transformer backbone as raw mode, but the diffusion
operates in flattened image space (H*W tokens instead of seq_len tokens).

Delay embedding:  ts(B,6,32) -> image(B,6,8,8) -> flat(B,6,64)  -> transformer
STFT embedding:   ts(B,6,32) -> image(B,12,8,9) -> flat(B,12,72) -> transformer
"""

from dataclasses import dataclass
from typing import Literal


@dataclass
class ImageEmbedConfig:
    """Image embedding hyperparameters."""
    embedding_type: Literal["delay", "stft"] = "delay"
    # Delay params (seq_len=32 -> 8x8 = 64 tokens, 6 channels)
    delay: int = 4
    embedding_dim: int = 8
    # STFT params (seq_len=32 -> 8x9 = 72 tokens, 12 channels)
    n_fft: int = 14
    hop_length: int = 4


@dataclass
class ModelConfig:
    """Transformer architecture. hidden_dim/num_layers overridable via CLI."""
    hidden_dim: int = 64
    num_layers: int = 3
    num_heads: int = 4        # 4 heads matches Diffusion-TS stock config
    ff_dim: int = 256         # 4x hidden_dim
    dropout: float = 0.1
    input_channels: int = 6   # raw ts channels (6 stock features)
    sequence_length: int = 32  # raw ts length — embedder computes image seq_len internally
    learnable_pos_enc: bool = True


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
    checkpoint_dir: str = "MyCode/output/checkpoints_image_delay"
    log_dir: str = "MyCode/output/logs_image"
    save_every_n_epochs: int = 250
    validate_every_n_epochs: int = 10


@dataclass
class SamplingConfig:
    sampler_type: Literal["ddpm", "ddim"] = "ddim"
    num_sampling_steps: int = 100
    eta: float = 0.0
    batch_size: int = 16
    output_dir: str = "MyCode/output/generated_samples_image"


@dataclass
class DataConfig:
    data_path: str = "MyCode/dataset/stocks_data.csv"
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    num_workers: int = 0
    pin_memory: bool = True
    neg_one_to_one: bool = True


class ImageTransformerConfig:
    """Combined config for image-transformer pipeline."""

    def __init__(self):
        self.image = ImageEmbedConfig()
        self.model = ModelConfig()
        self.diffusion = DiffusionConfig()
        self.training = TrainingConfig()
        self.sampling = SamplingConfig()
        self.data = DataConfig()

    def to_dict(self) -> dict:
        return {
            "image": self.image.__dict__,
            "model": self.model.__dict__,
            "diffusion": self.diffusion.__dict__,
            "training": self.training.__dict__,
            "sampling": self.sampling.__dict__,
            "data": self.data.__dict__,
        }
