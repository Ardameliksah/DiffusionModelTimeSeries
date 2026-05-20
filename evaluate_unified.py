"""
Unified evaluation script for both RAW and IMAGE modes.
Usage:
    python evaluate_unified.py --mode raw --checkpoint path/to/checkpoint.pt
    python evaluate_unified.py --mode image --checkpoint path/to/checkpoint.pt
"""

import torch
import argparse
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from config import Config, ImageVersionConfig
from models import create_model
from utils import compute_statistics, plot_real_vs_generated, set_seed


def evaluate(
    mode: str = "raw",
    checkpoint_path: str = None,
    device: str = "cpu",
    num_samples: int = 256,
    output_dir: str = None,
    n_metric_iterations: int = 5,
    compute_context_fid: bool = False,
    normalization: str = None,
    hidden_dim: int = None,
    num_layers: int = None,
    seed: int = 42,
    pos_enc: str = None,
    use_wandb: bool = False,
    wandb_project: str = "diffusion-timeseries",
):
    """
    Evaluate model in specified mode.
    
    Args:
        mode: "raw" or "image"
        checkpoint_path: Path to model checkpoint
        device: "cpu" or "cuda"
        num_samples: Number of samples to generate
        output_dir: Directory to save results
    """
    set_seed(seed)

    # Load config based on mode
    if mode == "raw":
        config = Config()
    elif mode == "image":
        config = ImageVersionConfig()
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    if normalization is not None:
        config.data.neg_one_to_one = (normalization == "minmax")
    if hidden_dim is not None:
        config.model.hidden_dim = hidden_dim
        config.model.ff_dim = hidden_dim * 4
    if num_layers is not None:
        config.model.num_layers = num_layers
    if pos_enc is not None:
        config.model.learnable_pos_enc = (pos_enc == "learnable")

    if output_dir is None:
        output_dir = config.sampling.output_dir
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print(f"EVALUATION - {mode.upper()} MODE")
    print("=" * 80)
    
    # Load model
    print("\n1. Loading model...")
    checkpoint = None
    if checkpoint_path and Path(checkpoint_path).exists():
        print(f"   Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        # Auto-restore arch from checkpoint if config was saved (overrides any CLI flags)
        saved_cfg = checkpoint.get("model_config")
        if saved_cfg is not None:
            for key, val in saved_cfg.items():
                if hasattr(config.model, key):
                    setattr(config.model, key, val)
            _pos = getattr(config.model, "learnable_pos_enc", "N/A (image mode)")
            print(f"   Arch restored from checkpoint: hidden_dim={config.model.hidden_dim}, "
                  f"num_layers={config.model.num_layers}, "
                  f"learnable_pos_enc={_pos}")

    model = create_model(config, model_type=mode, device=device)
    model.eval()

    if checkpoint is not None:
        model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    
    print(f"   Model type: {mode}")
    print(f"   Device: {device}")

    if use_wandb:
        import wandb
        _h = config.model.hidden_dim
        _l = config.model.num_layers
        wandb.init(
            project=wandb_project,
            name=f"eval_{mode}_h{_h}_l{_l}",
            group=f"{mode}_h{_h}_l{_l}",   # groups train+eval runs together in W&B
            job_type="eval",
            config={
                "mode": mode,
                "num_samples": num_samples,
                "n_metric_iterations": n_metric_iterations,
                "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
                "hidden_dim": _h,
                "num_layers": _l,
            },
        )

    # Generate samples
    print(f"\n3. Generating {num_samples} samples...")
    with torch.no_grad():
        samples = model.sample(
            batch_size=num_samples,
            sampler_type="ddim",
            num_steps=config.sampling.num_sampling_steps,
            eta=config.sampling.eta,
        )
    
    print(f"   Generated shape: {samples.shape}")
    samples_np = samples.cpu().numpy()
    
    # For comparison, load real data
    print("\n4. Loading real data...")
    from utils.data_utils import create_data_loaders
    
    _, test_loader, _ = create_data_loaders(
        csv_path=config.data.data_path,
        batch_size=num_samples,
        window_length=config.model.sequence_length,
        neg_one_to_one=config.data.neg_one_to_one,
        train_ratio=config.data.train_split,
        num_workers=0,
        per_window=config.data.per_window_norm,
    )
    
    real_data_batch = next(iter(test_loader)).cpu().numpy()
    print(f"   Real data shape: {real_data_batch.shape}")
    
    # Compute statistics
    print("\n5. Computing statistics...")
    real_stats = compute_statistics(real_data_batch)
    generated_stats = compute_statistics(samples_np)
    
    print("\n   Real Data Statistics:")
    for key, val in real_stats.items():
        if isinstance(val, (float, int)):
            print(f"     {key}: {val:.4f}")
    
    print("\n   Generated Data Statistics:")
    for key, val in generated_stats.items():
        if isinstance(val, (float, int)):
            print(f"     {key}: {val:.4f}")
    
    # Compute differences
    print("\n   Differences:")
    for key in real_stats:
        if isinstance(real_stats[key], (float, int)):
            diff = abs(real_stats[key] - generated_stats[key])
            print(f"     {key}: {diff:.4f}")
    
    # Create visualizations
    print("\n6. Creating visualizations...")
    save_path = Path(output_dir) / f"comparison_{mode}_mode.png"
    plot_real_vs_generated(
        real_data_batch,
        samples_np,
        save_path=str(save_path)
    )
    
    # Save samples
    sample_save_path = Path(output_dir) / f"samples_{mode}_mode.npy"
    np.save(sample_save_path, samples_np)
    print(f"   Samples saved: {sample_save_path}")
    print(f"   Visualization saved: {save_path}")
    
    # Save statistics
    h = config.model.hidden_dim
    l = config.model.num_layers
    stats_save_path = Path(output_dir) / f"eval_h{h}_l{l}_{mode}.txt"
    with open(stats_save_path, 'w') as f:
        f.write(f"Mode: {mode}\n")
        f.write(f"Checkpoint: {checkpoint_path}\n\n")
        f.write("Real Data Statistics:\n")
        for key, val in real_stats.items():
            f.write(f"  {key}: {val}\n")
        f.write("\nGenerated Data Statistics:\n")
        for key, val in generated_stats.items():
            f.write(f"  {key}: {val}\n")
    
    print(f"   Statistics saved: {stats_save_path}")

    # TimeGAN metrics
    print(f"\n7. Computing TimeGAN metrics ({n_metric_iterations} iterations each)...")
    from eval_metrics import evaluate_samples, vds_score, fdds_score, correlational_score

    # eval_metrics expects (N, seq_len, features) — transpose from (N, features, seq_len)
    real_for_metrics = real_data_batch.transpose(0, 2, 1)
    fake_for_metrics = samples_np.transpose(0, 2, 1)

    metric_results = evaluate_samples(
        real_for_metrics, fake_for_metrics,
        device=device,
        n_iterations=n_metric_iterations,
    )

    disc = metric_results["discriminative"]
    pred = metric_results["predictive"]
    print(f"\n   Discriminative score : {disc['mean']:.4f} +/- {disc['std']:.4f}  (target: 0.0)")
    print(f"   Test accuracy        : {disc['test_acc']:.4f}  (target: 0.5)")
    print(f"   Predictive MAE       : {pred['mean']:.4f} +/- {pred['std']:.4f}  (lower = better)")

    # Append metrics to the stats file
    with open(stats_save_path, 'a') as f:
        f.write("\nTimeGAN Metrics:\n")
        f.write(f"  discriminative_score: {disc['mean']:.4f} +/- {disc['std']:.4f}\n")
        f.write(f"  test_accuracy: {disc['test_acc']:.4f}\n")
        f.write(f"  predictive_mae: {pred['mean']:.4f} +/- {pred['std']:.4f}\n")
    print(f"   Metrics appended to: {stats_save_path}")

    # VDS, FDDS, Correlational Score
    print(f"\n8. Computing VDS / FDDS / Correlational Score...")
    vds   = vds_score(real_for_metrics, fake_for_metrics)
    fdds  = fdds_score(real_for_metrics, fake_for_metrics)
    corr  = correlational_score(real_for_metrics, fake_for_metrics)
    print(f"   VDS   : {vds:.4f}  (lower = better, KL divergence of value distributions)")
    print(f"   FDDS  : {fdds:.4f}  (lower = better, KL divergence of cross-corr distributions)")
    print(f"   Corr  : {corr:.4f}  (lower = better, |cacf_fake - cacf_real| / 10)")
    with open(stats_save_path, 'a') as f:
        f.write(f"\nPaD-TS / Diffusion-TS Metrics:\n")
        f.write(f"  vds: {vds:.4f}\n")
        f.write(f"  fdds: {fdds:.4f}\n")
        f.write(f"  correlational_score: {corr:.4f}\n")

    if compute_context_fid:
        print(f"\n9. Computing Context-FID score (TS2Vec encoder)...")
        import sys as _sys
        _diffts_path = str(Path(__file__).parent.parent / "Diffusion-TS")
        if _diffts_path not in _sys.path:
            _sys.path.insert(0, _diffts_path)
        from Utils.context_fid import Context_FID
        context_fid = Context_FID(real_for_metrics, fake_for_metrics)
        print(f"   Context-FID: {context_fid:.4f}  (lower = better)")
        with open(stats_save_path, 'a') as f:
            f.write(f"  context_fid: {context_fid:.4f}\n")
        print(f"   Metrics appended to: {stats_save_path}")

    if use_wandb:
        import wandb
        _eval_log = {
            "disc_score":          disc["mean"],
            "disc_score_std":      disc["std"],
            "test_acc":            disc["test_acc"],
            "pred_mae":            pred["mean"],
            "pred_mae_std":        pred["std"],
            "vds":                 vds,
            "fdds":                fdds,
            "correlational_score": corr,
            "comparison_plot":     wandb.Image(str(save_path)),
        }
        if compute_context_fid:
            _eval_log["context_fid"] = context_fid
        wandb.log(_eval_log)
        wandb.finish()
        print("   W&B run finished — metrics logged.")

    print("\n" + "=" * 80)
    print(f"Evaluation complete! ({mode} mode)")
    print(f"Results saved to: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model in raw or image mode")
    parser.add_argument(
        "--mode",
        choices=["raw", "image"],
        default="raw",
        help="Evaluation mode"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=256,
        help="Number of samples to generate"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="Number of independent runs for discriminative/predictive metric estimation (default: 5)"
    )
    parser.add_argument(
        "--context-fid",
        action="store_true",
        default=False,
        help="Compute Context-FID score using TS2Vec encoder (requires Diffusion-TS folder, slow)"
    )
    parser.add_argument(
        "--normalization",
        choices=["minmax", "zscore"],
        default=None,
        help="Normalization used during training (overrides config): minmax or zscore. Must match what was used at training time."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--pos-enc",
        choices=["learnable", "fixed"],
        default=None,
        help="Positional encoding type used at training time: learnable or fixed (must match checkpoint)"
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=None,
        help="Hidden dimension used during training (overrides config): 64=small, 128=medium, 256=large"
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=None,
        help="Number of transformer layers used during training (overrides config): 3=small, 6=medium, 8=large"
    )

    args = parser.parse_args()

    evaluate(
        mode=args.mode,
        checkpoint_path=args.checkpoint,
        device=args.device,
        num_samples=args.num_samples,
        output_dir=args.output_dir,
        n_metric_iterations=args.iterations,
        compute_context_fid=args.context_fid,
        normalization=args.normalization,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        seed=args.seed,
        pos_enc=args.pos_enc,
    )
