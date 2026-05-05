"""Utils module."""

import random
import warnings
import numpy as np
import torch


def set_seed(seed, cudnn_deterministic=False):
    if seed is not None:
        print(f"Global seed set to {seed}")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False

    if cudnn_deterministic:
        torch.backends.cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')


from .training import Trainer, create_optimizer_and_scheduler, TrainingLogger, save_checkpoint, load_checkpoint
from .sampling import DDPMSampler, DDIMSampler, sample_batch, create_sampler
from .visualization import plot_time_series, plot_real_vs_generated, compute_statistics, plot_statistics_comparison
from .image_transforms import TsImgEmbedder, DelayEmbedder, PatchEmbedder, STFTEmbedder, MRTIEmbedder
from . import persistence

__all__ = [
    "Trainer",
    "create_optimizer_and_scheduler",
    "TrainingLogger",
    "save_checkpoint",
    "load_checkpoint",
    "DDPMSampler",
    "DDIMSampler",
    "sample_batch",
    "create_sampler",
    "plot_time_series",
    "plot_real_vs_generated",
    "compute_statistics",
    "plot_statistics_comparison",
    "TsImgEmbedder",
    "DelayEmbedder",
    "PatchEmbedder",
    "STFTEmbedder",
    "MRTIEmbedder",
    "set_seed",
]
