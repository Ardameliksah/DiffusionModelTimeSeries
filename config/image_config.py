"""
Configuration for IMAGE-BASED Transformer Diffusion Model.
Use this config to train the model with time-series-to-image conversion via delay embedding.

Based on ImagenTime approach: converts 1D sequences to 2D images before processing.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

_ROOT = Path(__file__).parent.parent  # MyCode/


@dataclass
class ImageConfig:
    """Image transformation hyperparameters."""
    # Use "delay", "patch", "stft", or "mrti"
    embedding_type: Literal["delay", "patch", "stft", "mrti"] = "delay"
    
    # For delay embedding
    delay: int = 4  # Stride between windows
    embedding_dim: int = 8  # Height/width of square image
    
    # For patch embedding (alternative)
    patch_size: int = 4  # Size of each patch
    img_size: int = 8  # Target image size
    
    # For STFT embedding - tuned for seq_len=32 to produce 8x8 images
    n_fft: int = 14  # FFT window size (gives 8 freq bins)
    hop_length: int = 4  # Hop length (gives 5 time frames, pads to 8x8)
    
    # For MRTI embedding (Multi-Resolution Time Imaging)
    num_scales: int = 3  # Number of scales (M parameter)
    num_periods: int = 3  # Number of periods (K parameter)
    
    # Padding for reconstruction
    pad_to_square: bool = True
    pad_value: float = 0.0


@dataclass
class ModelConfig:
    """Transformer model hyperparameters (for image patches)."""
    hidden_dim: int = 128
    num_layers: int = 6
    num_heads: int = 8
    ff_dim: int = 512
    dropout: float = 0.1
    input_channels: int = 6  # Open, High, Low, Close, Adj_Close, Volume
    sequence_length: int = 32
    
    # These are COMPUTED from image config, don't change manually
    image_height: int = 8  # Will be updated from image_config.embedding_dim
    image_width: int = 8   # Will be updated from image_config.embedding_dim


@dataclass
class DiffusionConfig:
    """Gaussian diffusion process hyperparameters."""
    num_timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    noise_schedule: Literal["linear", "cosine", "exponential"] = "linear"
    variance_type: Literal["fixed_large", "fixed_small", "learned_range"] = "fixed_large"
    
    # For exponential schedule only
    gamma: float = 1.0


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    batch_size: int = 64
    learning_rate: float = 1e-3
    num_epochs: int = 500
    warmup_steps: int = 100
    weight_decay: float = 1e-4
    gradient_clip_val: float = 1.0
    
    # Learning rate scheduling
    lr_scheduler_type: Literal["cosine", "linear"] = "cosine"
    
    # Checkpoint and logging
    checkpoint_dir: str = str(_ROOT / "output" / "checkpoints_image")
    log_dir: str = str(_ROOT / "output" / "logs_image")
    save_every_n_epochs: int = 50
    validate_every_n_epochs: int = 10


@dataclass
class SamplingConfig:
    """Sampling hyperparameters."""
    sampler_type: Literal["ddpm", "ddim"] = "ddim"
    num_sampling_steps: int = 500
    eta: float = 0.0
    batch_size: int = 16
    output_dir: str = str(_ROOT / "output" / "generated_samples_image")


@dataclass
class DataConfig:
    """Data loading hyperparameters."""
    data_path: str = str(_ROOT / "dataset" / "stocks_data.csv")
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    num_workers: int = 0
    pin_memory: bool = True
    normalize: bool = True
    neg_one_to_one: bool = True  # MinMax [-1,1]


class ImageVersionConfig:
    """Combined configuration for IMAGE-BASED pipeline."""
    
    def __init__(self):
        self.image = ImageConfig()
        self.model = ModelConfig()
        self.diffusion = DiffusionConfig()
        self.training = TrainingConfig()
        self.sampling = SamplingConfig()
        self.data = DataConfig()
        
        # Auto-compute image dimensions
        self._update_image_dimensions()
    
    def _update_image_dimensions(self):
        """Update model image dimensions based on embedding type."""
        if self.image.embedding_type == "delay":
            self.model.image_height = self.image.embedding_dim
            self.model.image_width = self.image.embedding_dim
        else:  # patch
            self.model.image_height = self.image.img_size
            self.model.image_width = self.image.img_size
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary for logging."""
        return {
            "image": self.image.__dict__,
            "model": self.model.__dict__,
            "diffusion": self.diffusion.__dict__,
            "training": self.training.__dict__,
            "sampling": self.sampling.__dict__,
            "data": self.data.__dict__,
        }
