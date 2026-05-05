"""
Training script for Image-Transformer diffusion model.

Usage:
    python MyCode/src_image/train.py --embedding delay --device cuda
    python MyCode/src_image/train.py --embedding stft  --device cuda
    python MyCode/src_image/train.py --embedding delay --hidden-dim 64 --num-layers 3 --num-epochs 500
"""

import torch
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src_image.config import ImageTransformerConfig
from models.image_transformer_model import ImageTransformerDiffusionModel
from utils import Trainer, create_optimizer_and_scheduler, TrainingLogger, save_checkpoint, set_seed
from utils.data_utils import create_data_loaders


def train(
    embedding: str = "delay",
    device: str = "cpu",
    resume_from: str = None,
    num_epochs: int = None,
    batch_size: int = None,
    hidden_dim: int = None,
    num_layers: int = None,
    noise_schedule: str = None,
    checkpoint_dir: str = None,
    seed: int = 42,
):
    set_seed(seed)

    config = ImageTransformerConfig()
    config.image.embedding_type = embedding

    # CLI overrides
    if num_epochs is not None:
        config.training.num_epochs = num_epochs
    if batch_size is not None:
        config.training.batch_size = batch_size
    if hidden_dim is not None:
        config.model.hidden_dim = hidden_dim
        config.model.ff_dim = hidden_dim * 4
    if num_layers is not None:
        config.model.num_layers = num_layers
    if noise_schedule is not None:
        config.diffusion.noise_schedule = noise_schedule

    # Auto-set checkpoint dir based on embedding type and arch
    if checkpoint_dir is not None:
        config.training.checkpoint_dir = checkpoint_dir
    else:
        h = config.model.hidden_dim
        l = config.model.num_layers
        config.training.checkpoint_dir = str(Path(__file__).parent.parent / "output" / f"checkpoints_image_{embedding}_h{h}_l{l}")

    config.sampling.output_dir = str(Path(__file__).parent.parent / "output" / f"generated_samples_image_{embedding}")

    print("=" * 80)
    print(f"IMAGE-TRANSFORMER DIFFUSION — {embedding.upper()} EMBEDDING")
    print("=" * 80)
    print(config.to_dict())

    # Data
    print("\n1. Loading dataset...")
    train_loader, test_loader, _ = create_data_loaders(
        csv_path=config.data.data_path,
        batch_size=config.training.batch_size,
        window_length=config.model.sequence_length,
        neg_one_to_one=config.data.neg_one_to_one,
        train_ratio=config.data.train_split,
        num_workers=config.data.num_workers,
    )
    print(f"   Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")

    # Model
    print("\n2. Creating model...")
    model = ImageTransformerDiffusionModel(config, embedder_type=embedding, device=device)
    model.to(device)

    # Cache STFT normalization params from one training batch
    if embedding == "stft":
        print("   Caching STFT normalization parameters from training data...")
        for batch in train_loader:
            x_ts = batch.permute(0, 2, 1).to(device)
            model.embedder.cache_min_max_params(x_ts)
            break

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Parameters: {n_params:,}")
    print(f"   Embedding: {embedding}, Image: ({model.img_channels}, {model.H}, {model.W}), Seq: {model.img_seq_len}")

    # Optimizer
    print("\n3. Setting up optimizer...")
    optimizer, scheduler = create_optimizer_and_scheduler(
        model,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        num_epochs=config.training.num_epochs,
        warmup_steps=config.training.warmup_steps,
        scheduler_type=config.training.lr_scheduler_type,
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        gradient_clip_val=config.training.gradient_clip_val,
    )
    logger = TrainingLogger(config.training.log_dir)

    # Resume
    start_epoch = 0
    if resume_from and Path(resume_from).exists():
        print(f"\n4. Resuming from {resume_from}...")
        checkpoint = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint.get("epoch", 0)
        print(f"   Resumed from epoch {start_epoch}")

    # Save config for checkpoint restoration
    saved_model_config = {
        **config.model.__dict__,
        "embedder_type": embedding,
        "img_channels": model.img_channels,
        "img_H": model.H,
        "img_W": model.W,
        "img_seq_len": model.img_seq_len,
    }

    # Training loop
    print(f"\n5. Training ({config.training.num_epochs} epochs)...")
    best_val_loss = float("inf")

    for epoch in range(start_epoch, config.training.num_epochs):
        train_loss = trainer.train_epoch(train_loader)

        val_loss = None
        if epoch % config.training.validate_every_n_epochs == 0:
            val_loss = trainer.evaluate(test_loader)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    model, optimizer, epoch, val_loss,
                    checkpoint_dir=config.training.checkpoint_dir,
                    filename="best_model.pt",
                    model_config=saved_model_config,
                )

        current_lr = optimizer.param_groups[0]["lr"]
        logger.log_epoch(epoch, train_loss, val_loss, current_lr)

        if val_loss is not None:
            print(f"Epoch {epoch+1:3d} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")
        else:
            print(f"Epoch {epoch+1:3d} | Train: {train_loss:.4f}")

        if (epoch + 1) % config.training.save_every_n_epochs == 0:
            save_checkpoint(
                model, optimizer, epoch, train_loss,
                checkpoint_dir=config.training.checkpoint_dir,
                filename=f"checkpoint_epoch_{epoch+1}.pt",
                model_config=saved_model_config,
            )

    print("\n" + "=" * 80)
    print(f"Training complete! Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoints: {config.training.checkpoint_dir}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Image-Transformer diffusion model")
    parser.add_argument("--embedding", choices=["delay", "stft"], default="delay")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--noise-schedule", choices=["linear", "cosine", "exponential"], default=None)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train(
        embedding=args.embedding,
        device=args.device,
        resume_from=args.resume,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        noise_schedule=args.noise_schedule,
        checkpoint_dir=args.checkpoint_dir,
        seed=args.seed,
    )
