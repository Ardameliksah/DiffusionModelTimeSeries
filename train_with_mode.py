"""
Unified training script supporting both RAW and IMAGE-based diffusion models.
Usage:
    python train_with_mode.py --mode raw      (default, uses stocks_config)
    python train_with_mode.py --mode image    (uses image_config with delay embedding)
    python train_with_mode.py                 (defaults to raw mode)
"""

import torch
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from config.stocks_config import Config as RawConfig
from config.image_config import ImageVersionConfig
from models.model_selector import create_model
from utils import Trainer, create_optimizer_and_scheduler, TrainingLogger, save_checkpoint, set_seed
from utils.data_utils import create_data_loaders


def get_data_loaders(config):
    """Build train/test loaders."""
    norm_label = "MinMax [-1,1]" if config.data.neg_one_to_one else "Z-score"
    print(f"✓ Using {norm_label} normalization")
    train_loader, test_loader, dataset = create_data_loaders(
        csv_path=config.data.data_path,
        batch_size=config.training.batch_size,
        window_length=config.model.sequence_length,
        neg_one_to_one=config.data.neg_one_to_one,
        train_ratio=config.data.train_split,
        num_workers=config.data.num_workers,
        per_window=config.data.per_window_norm,
        pin_memory=config.data.pin_memory,
    )
    return train_loader, test_loader, dataset


def train(mode: str = "raw", device: str = "cpu", resume_from: str = None, embedding: str = "delay", num_epochs: int = None, batch_size: int = None, noise_schedule: str = None, checkpoint_dir: str = None, normalization: str = None, hidden_dim: int = None, num_layers: int = None, seed: int = 42, pos_enc: str = None, lr: float = None, num_workers: int = None):
    """
    Train the diffusion model in specified mode.

    Args:
        mode: "raw" or "image"
        device: "cpu" or "cuda"
        resume_from: Path to checkpoint to resume from
        embedding: "delay", "patch", "stft", or "mrti" (for image mode)
        num_epochs: Override num_epochs from config
        batch_size: Override batch_size from config
        noise_schedule: Override noise schedule ("linear", "cosine", "exponential")
        checkpoint_dir: Override checkpoint directory
        normalization: Override normalization ("minmax" = MinMax[-1,1], "zscore" = Z-score)
        hidden_dim: Override hidden dimension (64=small, 128=medium, 256=large)
        num_layers: Override number of transformer layers (3=small, 6=medium, 8=large)
    """
    set_seed(seed)

    # Select config based on mode
    if mode == "raw":
        config = RawConfig()
    elif mode == "image":
        config = ImageVersionConfig()
        # Set embedding type
        config.image.embedding_type = embedding
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    # Override config with CLI args if provided
    if num_epochs is not None:
        config.training.num_epochs = num_epochs
    if batch_size is not None:
        config.training.batch_size = batch_size
    if noise_schedule is not None:
        config.diffusion.noise_schedule = noise_schedule
    if checkpoint_dir is not None:
        config.training.checkpoint_dir = checkpoint_dir
    if normalization is not None:
        config.data.neg_one_to_one = (normalization == "minmax")
    if lr is not None:
        config.training.learning_rate = lr
    if hidden_dim is not None:
        config.model.hidden_dim = hidden_dim
        config.model.ff_dim = hidden_dim * 4  # keep standard 4x ratio
    if num_layers is not None:
        config.model.num_layers = num_layers
    if pos_enc is not None:
        config.model.learnable_pos_enc = (pos_enc == "learnable")
    if num_workers is not None:
        config.data.num_workers = num_workers

    print("=" * 80)
    print(f"TRANSFORMER DIFFUSION MODEL - {mode.upper()} MODE")
    print("=" * 80)
    print(config.to_dict())
    
    # Create data loaders
    print("\n1. Loading dataset...")
    train_loader, test_loader, dataset = get_data_loaders(config)
    print(f"   Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")
    
    # Create model
    print("\n2. Creating model...")
    model = create_model(config, model_type=mode, device=device)
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Mode: {mode}")
    print(f"   Noise schedule: {config.diffusion.noise_schedule}")
    
    if mode == "image":
        print(f"   Image embedding: {config.image.embedding_type}")
        print(f"   Image size: {config.image.embedding_dim}x{config.image.embedding_dim}")
    
    # Create optimizer and scheduler
    print("\n3. Setting up optimizer and scheduler...")
    actual_total_steps = len(train_loader) * config.training.num_epochs
    optimizer, scheduler = create_optimizer_and_scheduler(
        model,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        num_epochs=config.training.num_epochs,
        warmup_steps=config.training.warmup_steps,
        scheduler_type=config.training.lr_scheduler_type,
        total_steps=actual_total_steps,
    )
    print(f"   Total steps: {actual_total_steps} ({len(train_loader)} batches/epoch × {config.training.num_epochs} epochs)")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        gradient_clip_val=config.training.gradient_clip_val,
    )
    
    # Create logger
    logger = TrainingLogger(config.training.log_dir)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if resume_from and Path(resume_from).exists():
        print(f"\n4. Loading checkpoint from {resume_from}...")
        from utils import load_checkpoint
        checkpoint = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint.get("epoch", 0)
        print(f"   Resumed from epoch {start_epoch}")
    
    # Training loop
    print(f"\n5. Starting training ({mode} mode)...")
    print(f"   Epochs: {config.training.num_epochs}")
    print(f"   Batch size: {config.training.batch_size}")
    print()
    
    best_val_loss = float('inf')
    best_epoch = None
    saved_model_config = config.model.__dict__.copy()

    for epoch in range(start_epoch, config.training.num_epochs):
        # Training
        train_loss = trainer.train_epoch(train_loader)

        # Validation
        val_loss = None
        if epoch % config.training.validate_every_n_epochs == 0:
            val_loss = trainer.evaluate(test_loader)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                save_checkpoint(model, optimizer, epoch, val_loss,
                                checkpoint_dir=config.training.checkpoint_dir,
                                filename="best_model.pt",
                                model_config=saved_model_config)

        # Log metrics
        current_lr = optimizer.param_groups[0]['lr']
        logger.log_epoch(epoch, train_loss, val_loss, current_lr)

        if val_loss is not None:
            is_best = (epoch + 1) == best_epoch
            best_marker = f"  *** best model saved (epoch {best_epoch}) ***" if is_best else ""
            print(f"Epoch {epoch+1:3d} | Train: {train_loss:.4f} | Val: {val_loss:.4f}{best_marker}")
        else:
            print(f"Epoch {epoch+1:3d} | Train: {train_loss:.4f}")

        # Save checkpoint periodically
        if (epoch + 1) % config.training.save_every_n_epochs == 0:
            save_checkpoint(model, optimizer, epoch, train_loss,
                            checkpoint_dir=config.training.checkpoint_dir,
                            filename=f"checkpoint_epoch_{epoch+1}.pt",
                            model_config=saved_model_config)
            print(f"   Checkpoint saved: {config.training.checkpoint_dir}/checkpoint_epoch_{epoch+1}.pt")
    
    print("\n" + "=" * 80)
    print(f"Training complete! ({mode} mode)")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {config.training.checkpoint_dir}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train diffusion model in raw or image mode")
    parser.add_argument(
        "--mode",
        choices=["raw", "image"],
        default="raw",
        help="Training mode: 'raw' (direct TS) or 'image' (via delay embedding)"
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to train on"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--embedding",
        choices=["delay", "patch", "stft", "mrti"],
        default="delay",
        help="Image transformation method (image mode only): 'delay' (default), 'patch', 'stft', or 'mrti'"
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=None,
        help="Number of epochs (overrides config)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size (overrides config)"
    )
    parser.add_argument(
        "--noise-schedule",
        choices=["linear", "cosine", "exponential"],
        default=None,
        help="Noise schedule (overrides config): linear, cosine, or exponential"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Directory to save checkpoints (overrides config)"
    )
    parser.add_argument(
        "--normalization",
        choices=["minmax", "zscore"],
        default=None,
        help="Normalization method (overrides config): minmax = MinMax[-1,1], zscore = Z-score"
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
        help="Positional encoding type (overrides config): learnable (default, Diffusion-TS style) or fixed (sinusoidal)"
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=None,
        help="Hidden dimension (overrides config): 64=small, 128=medium, 256=large"
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=None,
        help="Number of transformer layers (overrides config): 3=small, 6=medium, 8=large"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate (overrides config), e.g. 1e-4"
    )

    args = parser.parse_args()

    print(f"Training mode: {args.mode}")
    print(f"Device: {args.device}")
    if args.mode == "image":
        print(f"Embedding: {args.embedding}")

    train(
        mode=args.mode,
        device=args.device,
        resume_from=args.resume,
        embedding=args.embedding,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        noise_schedule=args.noise_schedule,
        checkpoint_dir=args.checkpoint_dir,
        normalization=args.normalization,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        seed=args.seed,
        pos_enc=args.pos_enc,
        lr=args.lr,
    )
