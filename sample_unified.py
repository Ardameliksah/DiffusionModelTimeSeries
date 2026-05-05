"""
Unified Sampling Script - generates synthetic time series from both RAW and IMAGE models.
Works with any checkpoint from train_with_mode.py.

Usage:
    python sample_unified.py --mode raw --checkpoint path/to/model.pt --num-samples 100
    python sample_unified.py --mode image --embedding delay --checkpoint path/to/model.pt --num-samples 100
"""

import torch
import argparse
import json
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from config.stocks_config import Config as RawConfig
from config.image_config import ImageVersionConfig
from models.model_selector import create_model, UnifiedDiffusionModel
from utils.data_utils import StockDataset, build_scaler
from utils import compute_statistics


def load_model(checkpoint_path, mode, config, device):
    """Load trained model."""
    model = create_model(config, mode, device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model


def generate(checkpoint_path, mode, config, num_samples=100, num_steps=50, eta=0.0, device='cpu'):
    """Generate synthetic samples."""
    batch_size = config.sampling.batch_size
    num_batches = (num_samples + batch_size - 1) // batch_size

    print(f"\n{'='*80}")
    print(f"SAMPLING - {mode.upper()} MODE")
    print(f"{'='*80}")
    print(f"Model: {checkpoint_path}")
    print(f"Samples: {num_samples}")
    print(f"Batch Size: {batch_size}")
    print(f"Sampling Steps: {num_steps}")
    print(f"Device: {device}")

    # Load model
    print(f"\nLoading model...")
    model = load_model(checkpoint_path, mode, config, device)
    print(f"✓ Model loaded (parameters: {sum(p.numel() for p in model.parameters()):,})")

    # Generate samples using the model's built-in sampler (same path as evaluate_unified.py)
    print(f"\nGenerating {num_samples} samples in {num_batches} batches...")
    all_samples = []

    for batch_idx in range(num_batches):
        batch_size_curr = min(batch_size, num_samples - batch_idx * batch_size)

        samples = model.sample(
            batch_size=batch_size_curr,
            sampler_type="ddim",
            num_steps=num_steps,
            eta=eta,
        )

        all_samples.append(samples.cpu().numpy())
        print(f"  Generated batch {batch_idx + 1}/{num_batches} ({batch_size_curr} samples)")
    
    all_samples = np.concatenate(all_samples, axis=0)[:num_samples]
    print(f"\n✓ Generated shape: {all_samples.shape}")
    
    return all_samples


def denormalize(samples, dataset):
    """Denormalize samples to original scale."""
    return dataset.denormalize(samples)


def save_samples(samples, output_dir, mode, embedding=None):
    """Save generated samples."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate filename
    if embedding:
        filename = f"samples_{mode}_{embedding}.npz"
    else:
        filename = f"samples_{mode}.npz"
    
    filepath = output_dir / filename
    
    np.savez(filepath, samples=samples)
    print(f"\n✓ Samples saved to: {filepath}")
    print(f"  Shape: {samples.shape}")
    print(f"  File size: {filepath.stat().st_size / 1e6:.2f} MB")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic time series samples")
    parser.add_argument("--mode", choices=["raw", "image"], default="raw", help="Model mode")
    parser.add_argument("--embedding", choices=["delay", "patch", "stft", "mrti"], default="delay", help="Image embedding type (for image mode)")
    parser.add_argument("--checkpoint", type=str, default=str(Path(__file__).parent / "output" / "checkpoints" / "best_model.pt"), help="Path to checkpoint")
    parser.add_argument("--num-samples", type=int, default=100, help="Number of samples to generate")
    parser.add_argument("--num-steps", type=int, default=50, help="DDIM sampling steps")
    parser.add_argument("--eta", type=float, default=0.0, help="DDIM eta parameter")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda", help="Device")
    
    args = parser.parse_args()
    
    # Load config based on mode
    print(f"\n{'='*80}")
    print(f"LOADING CONFIG - {args.mode.upper()} MODE")
    print(f"{'='*80}")
    
    if args.mode == "raw":
        config = RawConfig()
        print(f"✓ Using RawConfig (stocks_config.py)")
        checkpoint_path = args.checkpoint
    else:
        config = ImageVersionConfig()
        config.image.embedding_type = args.embedding
        print(f"✓ Using ImageVersionConfig with {args.embedding} embedding")
        # Auto-detect checkpoint for image mode
        if args.checkpoint is None:
            checkpoint_path = str(Path(__file__).parent / "output" / "checkpoints_image" / "checkpoint_epoch_0010.pt")
            print(f"✓ Auto-detected checkpoint: {checkpoint_path}")
        else:
            checkpoint_path = args.checkpoint
    
    print(f"Config details:")
    print(f"  Sequence length: {config.model.sequence_length}")
    print(f"  Channels: {config.model.input_channels}")
    print(f"  Batch size: {config.sampling.batch_size}")
    
    # Generate samples
    samples = generate(
        checkpoint_path=checkpoint_path,
        mode=args.mode,
        config=config,
        num_samples=args.num_samples,
        num_steps=args.num_steps,
        eta=args.eta,
        device=args.device
    )
    
    # Load dataset for denormalization
    print(f"\nDenormalizing samples...")
    dataset = StockDataset(
        csv_path=config.data.data_path,
        scaler=build_scaler(config.data.neg_one_to_one),
        window_length=config.model.sequence_length,
    )
    samples_denorm = denormalize(samples, dataset)
    print(f"✓ Denormalized shape: {samples_denorm.shape}")
    
    # Save
    output_dir = config.sampling.output_dir
    save_samples(samples_denorm, output_dir, args.mode, args.embedding if args.mode == "image" else None)
    
    # Compute statistics
    print(f"\nComputing statistics...")
    mean = samples_denorm.mean(axis=(0, 2))
    std = samples_denorm.std(axis=(0, 2))
    print(f"Mean per channel: {mean}")
    print(f"Std per channel: {std}")
    
    print(f"\n{'='*80}")
    print(f"SAMPLING COMPLETE")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
