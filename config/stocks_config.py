"""
Configuration file for Transformer-based Diffusion Model for Time Series.
Centralizes all hyperparameters: model architecture, diffusion process, and training settings.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

_ROOT = Path(__file__).parent.parent  # MyCode/


@dataclass
class ModelConfig:
    """Transformer model hyperparameters."""
    hidden_dim: int = 128
    num_layers: int = 6
    num_heads: int = 8
    ff_dim: int = 512
    dropout: float = 0.1
    input_channels: int = 6  # Open, High, Low, Close, Adj_Close, Volume
    sequence_length: int = 32
    learnable_pos_enc: bool = True


@dataclass
class DiffusionConfig:
    """Gaussian diffusion process hyperparameters."""
    num_timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    noise_schedule: Literal["linear", "cosine", "exponential"] = "cosine"
    variance_type: Literal["fixed_large", "fixed_small", "learned_range"] = "fixed_large"
    
    # For exponential schedule only
    gamma: float = 1.0


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    batch_size: int = 64
    learning_rate: float = 5e-4
    num_epochs: int = 500
    warmup_steps: int = 100
    weight_decay: float = 1e-4
    gradient_clip_val: float = 1.0
    
    # Learning rate scheduling
    lr_scheduler_type: Literal["cosine", "linear"] = "cosine"
    
    # Checkpoint and logging
    checkpoint_dir: str = str(_ROOT / "output" / "checkpoints")
    log_dir: str = str(_ROOT / "output" / "logs")
    save_every_n_epochs: int = 250
    validate_every_n_epochs: int = 10


@dataclass
class SamplingConfig:
    """Sampling hyperparameters."""
    sampler_type: Literal["ddpm", "ddim"] = "ddim"
    num_sampling_steps: int = 500  # For DDIM; DDPM uses full num_timesteps
    eta: float = 0.0  # Controls stochasticity in DDIM (0 = deterministic)
    batch_size: int = 16
    output_dir: str = str(_ROOT / "output" / "generated_samples")


@dataclass
class DataConfig:
    """Data loading hyperparameters."""
    data_path: str = str(_ROOT / "dataset" / "stocks_data.csv")
    train_split: float = 1.0   # 1.0 = all data for training (DiffusionTS protocol)
    val_split: float = 0.0
    test_split: float = 0.0
    num_workers: int = 0  # Set to 0 on Windows to avoid multiprocessing overhead
    pin_memory: bool = True
    normalize: bool = True
    # SINGLE parameter determines normalization method:
    # - True: MinMax + [-1,1] remap (Diffusion-TS, BOUNDED, RECOMMENDED) + TEMPORAL SPLIT
    # - False: Z-score (Original, unbounded) + RANDOM SPLIT
    neg_one_to_one: bool = True
    per_window_norm: bool = True   # True = per-window MinMax; False = global scaler


@dataclass
class DecompositionConfig:
    """Optional decomposition loss weights. 0.0 = disabled (no effect on loss)."""
    fft_weight:    float = 0.0  # FFT frequency-domain auxiliary loss
    trend_weight:  float = 0.0  # Moving-average trend component loss
    season_weight: float = 0.0  # Seasonal residual component loss
    trend_kernel:  int   = 5    # Avg-pool kernel size for trend extraction


class Config:
    """Combined configuration for the entire pipeline."""
    
    def __init__(self):
        self.model = ModelConfig()
        self.diffusion = DiffusionConfig()
        self.training = TrainingConfig()
        self.sampling = SamplingConfig()
        self.data = DataConfig()
        self.decomposition = DecompositionConfig()
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary for logging."""
        return {
            "model": self.model.__dict__,
            "diffusion": self.diffusion.__dict__,
            "training": self.training.__dict__,
            "sampling": self.sampling.__dict__,
            "data": self.data.__dict__,
            "decomposition": self.decomposition.__dict__,
        }
    
    def __repr__(self) -> str:
        """Pretty print configuration."""
        config_dict = self.to_dict()
        config_str = "Configuration:\n"
        for section, values in config_dict.items():
            config_str += f"\n{section.upper()}:\n"
            for key, val in values.items():
                config_str += f"  {key}: {val}\n"
        return config_str


# Default configuration instance
config = Config()
