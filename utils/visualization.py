"""
Visualization utilities for diffusion model results.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, List, Tuple
import seaborn as sns

sns.set_style("whitegrid")


def plot_time_series(
    sample: np.ndarray,
    title: str = "Time Series",
    feature_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (15, 10),
) -> plt.Figure:
    """
    Plot a single time series with multiple features.
    
    Args:
        sample: Array of shape (num_features, seq_len)
        title: Title for the plot
        feature_names: Names of features (default: Feature_0, Feature_1, ...)
        figsize: Figure size
    
    Returns:
        Matplotlib figure object
    """
    num_features = sample.shape[0]
    
    if feature_names is None:
        feature_names = [f"Feature_{i}" for i in range(num_features)]
    
    fig, axes = plt.subplots(num_features, 1, figsize=figsize, sharex=True)
    if num_features == 1:
        axes = [axes]
    
    for i, (ax, feature_name) in enumerate(zip(axes, feature_names)):
        ax.plot(sample[i], linewidth=1.5, color="steelblue")
        ax.set_ylabel(feature_name, fontsize=10)
        ax.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel("Timestep", fontsize=10)
    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    
    return fig


def plot_real_vs_generated(
    real_batch: torch.Tensor,
    generated_batch: torch.Tensor,
    num_samples: int = 4,
    feature_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
):
    """
    Plot real and generated samples side by side.
    
    Args:
        real_batch: Real samples (batch_size, num_features, seq_len)
        generated_batch: Generated samples (batch_size, num_features, seq_len)
        num_samples: Number of samples to visualize
        feature_names: Names of features
        save_path: Path to save figure
    """
    num_samples = min(num_samples, real_batch.shape[0], generated_batch.shape[0])
    num_features = real_batch.shape[1]
    
    if feature_names is None:
        feature_names = ["Open", "High", "Low", "Close", "Adj_Close", "Volume"][:num_features]
    
    fig, axes = plt.subplots(
        num_samples,
        num_features,
        figsize=(5 * num_features, 3 * num_samples),
    )
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    import numpy as np
    real_np = real_batch if isinstance(real_batch, np.ndarray) else real_batch.detach().cpu().numpy()
    gen_np = generated_batch if isinstance(generated_batch, np.ndarray) else generated_batch.detach().cpu().numpy()
    
    for sample_idx in range(num_samples):
        for feature_idx, feature_name in enumerate(feature_names):
            ax = axes[sample_idx, feature_idx]
            
            # Plot real and generated
            ax.plot(real_np[sample_idx, feature_idx], label="Real", linewidth=2, alpha=0.7)
            ax.plot(gen_np[sample_idx, feature_idx], label="Generated", linewidth=2, alpha=0.7)
            
            if sample_idx == 0:
                ax.set_title(feature_name, fontsize=11, fontweight="bold")
            if feature_idx == 0:
                ax.set_ylabel(f"Sample {sample_idx}", fontsize=10)
            if sample_idx == num_samples - 1:
                ax.set_xlabel("Timestep", fontsize=9)
            
            ax.grid(True, alpha=0.3)
            if sample_idx == 0 and feature_idx == 0:
                ax.legend(loc="upper right")
    
    plt.tight_layout()
    
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {save_path}")
    
    return fig


def compute_statistics(
    samples: torch.Tensor,
    feature_names: Optional[List[str]] = None,
) -> dict:
    """
    Compute statistics for generated samples.
    
    Args:
        samples: Samples of shape (batch_size, num_features, seq_len)
        feature_names: Names of features
    
    Returns:
        Dictionary with statistics
    """
    num_features = samples.shape[1]
    
    if feature_names is None:
        feature_names = [f"Feature_{i}" for i in range(num_features)]
    
    stats = {}
    
    samples_np = np.asarray(samples) if not isinstance(samples, np.ndarray) else samples
    
    for i, feature_name in enumerate(feature_names):
        feature_data = samples_np[:, i, :].flatten()
        
        stats[feature_name] = {
            "mean": float(np.mean(feature_data)),
            "std": float(np.std(feature_data)),
            "min": float(np.min(feature_data)),
            "max": float(np.max(feature_data)),
            "median": float(np.median(feature_data)),
        }
    
    return stats


def plot_statistics_comparison(
    real_stats: dict,
    generated_stats: dict,
    save_path: Optional[str] = None,
):
    """
    Plot comparison of statistics between real and generated data.
    
    Args:
        real_stats: Statistics for real data
        generated_stats: Statistics for generated data
        save_path: Path to save figure
    """
    metrics = ["mean", "std", "min", "max"]
    features = list(real_stats.keys())
    
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4))
    
    for metric_idx, metric in enumerate(metrics):
        ax = axes[metric_idx]
        
        real_vals = [real_stats[f][metric] for f in features]
        gen_vals = [generated_stats[f][metric] for f in features]
        
        x = np.arange(len(features))
        width = 0.35
        
        ax.bar(x - width / 2, real_vals, width, label="Real", alpha=0.8)
        ax.bar(x + width / 2, gen_vals, width, label="Generated", alpha=0.8)
        
        ax.set_ylabel(metric.capitalize(), fontsize=11)
        ax.set_title(f"{metric.upper()}", fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(features, rotation=45, ha="right")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {save_path}")
    
    return fig


from typing import Tuple


def plot_marginal_densities(
    real_data,
    fake_data,
    feature_names=None,
    save_path=None,
):
    """
    Per-feature KDE probability density overlay: real (blue) vs generated (orange).
    Matches the WaveletDiff paper evaluation visualisation style.

    Args:
        real_data:     (N, C, L) channels-first array
        fake_data:     (N, C, L) channels-first array
        feature_names: list of strings (default: Open/High/Low/Close/Adj_Close/Volume)
        save_path:     file path to save the figure

    Returns:
        matplotlib Figure
    """
    from scipy.stats import gaussian_kde

    real_np = np.asarray(real_data)
    fake_np = np.asarray(fake_data)
    C = real_np.shape[1]

    if feature_names is None:
        feature_names = ["Open", "High", "Low", "Close", "Adj_Close", "Volume"][:C]

    fig, axes = plt.subplots(1, C, figsize=(4 * C, 4))
    if C == 1:
        axes = [axes]

    for i, (ax, name) in enumerate(zip(axes, feature_names)):
        r = real_np[:, i, :].flatten()
        f = fake_np[:, i, :].flatten()

        lo = min(r.min(), f.min())
        hi = max(r.max(), f.max())
        xs = np.linspace(lo, hi, 300)

        try:
            kde_r = gaussian_kde(r)
            kde_f = gaussian_kde(f)
        except Exception:
            ax.set_title(f"{name} (KDE failed)", fontsize=10)
            continue

        ax.plot(xs, kde_r(xs), color="steelblue",  lw=2, label="Real")
        ax.plot(xs, kde_f(xs), color="darkorange", lw=2, label="Generated")
        ax.fill_between(xs, kde_r(xs), alpha=0.15, color="steelblue")
        ax.fill_between(xs, kde_f(xs), alpha=0.15, color="darkorange")

        ax.set_title(name, fontsize=11, fontweight="bold")
        ax.set_xlabel("Value", fontsize=9)
        ax.set_ylabel("Density", fontsize=9)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=9)

    plt.suptitle("Marginal Probability Densities — Real vs Generated", fontsize=13)
    plt.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"   Density plot saved: {save_path}")

    return fig
