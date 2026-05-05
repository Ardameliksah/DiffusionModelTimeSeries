"""
Evaluation script for Image-Transformer diffusion model.

Usage:
    python MyCode/src_image/evaluate.py --embedding delay --checkpoint MyCode/output/checkpoints_image_delay_h64_l3/best_model.pt
    python MyCode/src_image/evaluate.py --embedding stft  --checkpoint MyCode/output/checkpoints_image_stft_h64_l3/best_model.pt
"""

import torch
import argparse
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src_image.config import ImageTransformerConfig
from models.image_transformer_model import ImageTransformerDiffusionModel
from utils import compute_statistics, plot_real_vs_generated, set_seed
from utils.data_utils import create_data_loaders


def evaluate(
    embedding: str = "delay",
    checkpoint_path: str = None,
    device: str = "cpu",
    num_samples: int = 100,
    output_dir: str = None,
    n_metric_iterations: int = 5,
    seed: int = 42,
):
    set_seed(seed)

    config = ImageTransformerConfig()
    config.image.embedding_type = embedding

    if output_dir is None:
        output_dir = f"MyCode/output/generated_samples_image_{embedding}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"EVALUATION — IMAGE-TRANSFORMER ({embedding.upper()})")
    print("=" * 80)

    # Load checkpoint and auto-restore arch
    print("\n1. Loading model...")
    checkpoint = None
    if checkpoint_path and Path(checkpoint_path).exists():
        print(f"   Checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        saved_cfg = checkpoint.get("model_config")
        if saved_cfg is not None:
            for key in ["hidden_dim", "num_layers", "num_heads", "ff_dim", "dropout", "learnable_pos_enc"]:
                if key in saved_cfg:
                    setattr(config.model, key, saved_cfg[key])
            if "embedder_type" in saved_cfg:
                embedding = saved_cfg["embedder_type"]
                config.image.embedding_type = embedding
            print(f"   Arch restored: hidden_dim={config.model.hidden_dim}, "
                  f"num_layers={config.model.num_layers}, embedding={embedding}")

    model = ImageTransformerDiffusionModel(config, embedder_type=embedding, device=device)
    model.to(device)
    model.eval()

    if checkpoint is not None:
        model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
        print(f"   Image space: ({model.img_channels}, {model.H}, {model.W}), seq={model.img_seq_len}")

    # Data
    print("\n2. Loading data...")
    train_loader, test_loader, _ = create_data_loaders(
        csv_path=config.data.data_path,
        batch_size=num_samples,
        window_length=config.model.sequence_length,
        neg_one_to_one=config.data.neg_one_to_one,
        train_ratio=config.data.train_split,
        num_workers=0,
    )

    # Cache STFT normalization params (must match training)
    if embedding == "stft":
        for batch in train_loader:
            x_ts = batch.permute(0, 2, 1).to(device)
            model.embedder.cache_min_max_params(x_ts)
            break

    real_data_batch = next(iter(test_loader)).cpu().numpy()
    real_train_batch = next(iter(train_loader)).cpu().numpy()
    print(f"   Real data shape: {real_data_batch.shape}")

    # Generate
    print(f"\n3. Generating {num_samples} samples...")
    with torch.no_grad():
        samples = model.sample(
            batch_size=num_samples,
            sampler_type="ddim",
            num_steps=config.sampling.num_sampling_steps,
            eta=config.sampling.eta,
        )
    samples_np = samples.cpu().numpy()
    print(f"   Generated shape: {samples_np.shape}")

    # Statistics
    print("\n4. Computing statistics...")
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

    # Visualization (use training data for fair visual comparison)
    print("\n5. Creating visualizations...")
    save_path = Path(output_dir) / f"comparison_image_{embedding}.png"
    plot_real_vs_generated(real_train_batch, samples_np, save_path=str(save_path))

    # Save samples
    np.save(Path(output_dir) / f"samples_image_{embedding}.npy", samples_np)

    # Save stats
    stats_path = Path(output_dir) / f"statistics_image_{embedding}.txt"
    with open(stats_path, "w") as f:
        f.write(f"Mode: image_transformer ({embedding})\n")
        f.write(f"Checkpoint: {checkpoint_path}\n\n")
        f.write("Real Data Statistics:\n")
        for key, val in real_stats.items():
            f.write(f"  {key}: {val}\n")
        f.write("\nGenerated Data Statistics:\n")
        for key, val in generated_stats.items():
            f.write(f"  {key}: {val}\n")
    print(f"   Stats: {stats_path}")
    print(f"   Visualization: {save_path}")

    # TimeGAN metrics
    print(f"\n6. Computing TimeGAN metrics ({n_metric_iterations} iterations)...")
    from eval_metrics import evaluate_samples, vds_score, fdds_score, correlational_score

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

    vds   = vds_score(real_for_metrics, fake_for_metrics)
    fdds  = fdds_score(real_for_metrics, fake_for_metrics)
    corr  = correlational_score(real_for_metrics, fake_for_metrics)
    print(f"\n   VDS   : {vds:.4f}")
    print(f"   FDDS  : {fdds:.4f}")
    print(f"   Corr  : {corr:.4f}")

    with open(stats_path, "a") as f:
        f.write("\nTimeGAN Metrics:\n")
        f.write(f"  discriminative_score: {disc['mean']:.4f} +/- {disc['std']:.4f}\n")
        f.write(f"  test_accuracy: {disc['test_acc']:.4f}\n")
        f.write(f"  predictive_mae: {pred['mean']:.4f} +/- {pred['std']:.4f}\n")
        f.write(f"\nPaD-TS / Diffusion-TS Metrics:\n")
        f.write(f"  vds: {vds:.4f}\n")
        f.write(f"  fdds: {fdds:.4f}\n")
        f.write(f"  correlational_score: {corr:.4f}\n")

    print("\n" + "=" * 80)
    print(f"Evaluation complete! Results: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Image-Transformer diffusion model")
    parser.add_argument("--embedding", choices=["delay", "stft"], default="delay")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    evaluate(
        embedding=args.embedding,
        checkpoint_path=args.checkpoint,
        device=args.device,
        num_samples=args.num_samples,
        output_dir=args.output_dir,
        n_metric_iterations=args.iterations,
        seed=args.seed,
    )
