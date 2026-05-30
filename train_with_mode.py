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


def _compute_inline_metrics(model, test_loader, device, n_iterations, train_loader=None):
    """
    Generate samples and compute discriminative / predictive / VDS / FDDS / corr
    against real test data. Called mid-training at validation checkpoints.

    Returns a dict of metric values, or None if eval_metrics is unavailable.
    """
    import numpy as np
    try:
        from eval_metrics import evaluate_samples, vds_score, fdds_score, correlational_score
    except ImportError:
        print("   [metrics] eval_metrics module not found — skipping inline metrics.")
        return None

    model.eval()

    # ── Collect ALL real data (DiffusionTS protocol: full dataset, no cap) ────
    real_batches = []
    with torch.no_grad():
        for batch in test_loader:
            real_batches.append(batch.cpu().numpy())
    if not real_batches:
        model.train()
        return None

    real_np = np.concatenate(real_batches, axis=0)    # (N, C, L) — full dataset
    n = real_np.shape[0]

    # ── Generate same number of fake samples ───────────────────────────────
    with torch.no_grad():
        fake_np = model.sample(batch_size=n, sampler_type="ddim",
                               num_steps=50, eta=0.0).cpu().numpy()   # (N, C, L)

    # eval_metrics expects (N, L, C)
    real_m = real_np.transpose(0, 2, 1)
    fake_m = fake_np.transpose(0, 2, 1)

    # ── Compute all metrics ─────────────────────────────────────────────────
    # disc_iterations=500 / pred_iterations=1000 are fast inline versions.
    # Full evaluation (disc=2000, pred=5000) is done separately after training.
    try:
        results = evaluate_samples(real_m, fake_m, device=device, n_iterations=n_iterations,
                                   disc_iterations=2000, pred_iterations=5000)
        vds  = vds_score(real_m, fake_m)
        fdds = fdds_score(real_m, fake_m)
        corr = correlational_score(real_m, fake_m)
    except Exception as exc:
        import traceback
        print(f"   [metrics] Error during metric computation: {exc}")
        traceback.print_exc()
        model.train()
        return None

    disc = results["discriminative"]
    pred = results["predictive"]

    # ── Context-FID — matches DiffusionTS protocol: use all training data ───────
    # disc/pred/vds/fdds use num_samples (fast); Context-FID needs N > repr_dim=320
    # and must use train data to match the DiffusionTS paper protocol.
    # TS2Vec uses cached weights (pretrain_ts2vec.py) — no retraining here.
    context_fid = None
    _cfid_loader = train_loader if train_loader is not None else test_loader
    try:
        from utils.context_fid import Context_FID
        _ts2vec_device = "cuda" if device == "cuda" else "cpu"

        all_batches = [_b.cpu().numpy() for _b in _cfid_loader]
        all_real_np = np.concatenate(all_batches, axis=0)      # (N, C, L)
        all_real_m  = all_real_np.transpose(0, 2, 1)           # (N, L, C)

        with torch.no_grad():
            all_fake_np = model.sample(
                batch_size=all_real_np.shape[0],
                sampler_type="ddim", num_steps=50, eta=0.0,
            ).cpu().numpy()
        all_fake_m = all_fake_np.transpose(0, 2, 1)

        context_fid = Context_FID(all_real_m, all_fake_m, device=_ts2vec_device)
        print(f"   Context-FID: {context_fid:.4f}  (N={all_real_np.shape[0]})")
    except Exception as exc:
        print(f"   [metrics] Context-FID failed (skipping): {exc}")

    model.train()
    out = {
        "disc_score":          disc["mean"],
        "disc_score_std":      disc["std"],
        "test_acc":            disc["test_acc"],
        "pred_mae":            pred["mean"],
        "pred_mae_std":        pred["std"],
        "vds":                 vds,
        "fdds":                fdds,
        "correlational_score": corr,
    }
    if context_fid is not None:
        out["context_fid"] = context_fid
    return out


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


def train(mode: str = "raw", device: str = "cpu", resume_from: str = None, embedding: str = "delay", num_epochs: int = None, batch_size: int = None, noise_schedule: str = None, checkpoint_dir: str = None, normalization: str = None, hidden_dim: int = None, num_layers: int = None, seed: int = 42, pos_enc: str = None, lr: float = None, num_workers: int = None, use_wandb: bool = False, wandb_project: str = "diffusion-timeseries", wandb_run_name: str = None, wandb_group: str = None, eval_metrics: bool = False, eval_metrics_every: int = 100, n_metric_iterations: int = 3, img_pred_objective: str = None, img_loss_type: str = None, fft_weight: float = None, trend_weight: float = None, season_weight: float = None):
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
    if mode in ("raw", "decomposition"):
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
    # Image-mode specific overrides (no-ops for raw mode)
    if img_pred_objective is not None and hasattr(config.model, 'pred_objective'):
        config.model.pred_objective = img_pred_objective
    if img_loss_type is not None and hasattr(config.model, 'loss_type'):
        config.model.loss_type = img_loss_type

    # Decomposition loss weights
    # decomposition mode: sensible defaults; raw mode: only apply if explicitly passed
    if mode == "decomposition":
        config.decomposition.fft_weight    = fft_weight    if fft_weight    is not None else 0.1
        config.decomposition.trend_weight  = trend_weight  if trend_weight  is not None else 0.5
        config.decomposition.season_weight = season_weight if season_weight is not None else 0.0
    else:
        if hasattr(config, 'decomposition'):
            if fft_weight    is not None: config.decomposition.fft_weight    = fft_weight
            if trend_weight  is not None: config.decomposition.trend_weight  = trend_weight
            if season_weight is not None: config.decomposition.season_weight = season_weight

    if use_wandb:
        import wandb
        _h = config.model.hidden_dim
        _l = config.model.num_layers
        wandb.init(
            project=wandb_project,
            name=wandb_run_name or f"train_{mode}_h{_h}_l{_l}",
            group=wandb_group or f"{mode}_h{_h}_l{_l}",
            job_type="train",
            config=config.to_dict(),
        )

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
    if hasattr(config, 'decomposition') and (mode == "decomposition" or any([
        config.decomposition.fft_weight, config.decomposition.trend_weight,
        config.decomposition.season_weight,
    ])):
        d = config.decomposition
        print(f"   Decomposition loss: fft={d.fft_weight}  trend={d.trend_weight}  "
              f"season={d.season_weight}  kernel={d.trend_kernel}")
    
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
                if use_wandb:
                    wandb.run.summary["best_val_loss"] = best_val_loss
                    wandb.run.summary["best_epoch"] = best_epoch

        # ── Inline evaluation metrics (optional, slow) ───────────────────────
        inline_metrics = None
        if eval_metrics and (epoch + 1) % eval_metrics_every == 0:
            print(f"   Computing inline metrics  "
                  f"(every {eval_metrics_every} epochs | "
                  f"n_iter={n_metric_iterations} | full dataset) ...")
            inline_metrics = _compute_inline_metrics(
                model, test_loader, device,
                n_iterations=n_metric_iterations,
                train_loader=train_loader,
            )
            if inline_metrics is not None:
                print(
                    f"   DiscScore: {inline_metrics['disc_score']:.4f} "
                    f"± {inline_metrics['disc_score_std']:.4f}  "
                    f"(acc={inline_metrics['test_acc']:.4f})  |  "
                    f"PredMAE: {inline_metrics['pred_mae']:.4f}  |  "
                    f"VDS: {inline_metrics['vds']:.4f}  |  "
                    f"FDDS: {inline_metrics['fdds']:.4f}  |  "
                    f"Corr: {inline_metrics['correlational_score']:.4f}"
                )
        # ─────────────────────────────────────────────────────────────────────

        # Log metrics
        current_lr = optimizer.param_groups[0]['lr']
        logger.log_epoch(epoch, train_loss, val_loss, current_lr)

        if use_wandb:
            _log = {"epoch": epoch + 1, "train_loss": train_loss, "lr": current_lr}
            if val_loss is not None:
                _log["val_loss"] = val_loss
            if inline_metrics is not None:
                _log.update(inline_metrics)
            wandb.log(_log)

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
    
    # ── Loss curve plot ───────────────────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs_all  = [e + 1 for e in logger.metrics["epoch"]]
    train_losses = logger.metrics["train_loss"]

    val_epochs  = [e + 1 for e, v in zip(logger.metrics["epoch"], logger.metrics["val_loss"]) if v is not None]
    val_losses  = [v for v in logger.metrics["val_loss"] if v is not None]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(epochs_all, train_losses, label="Train loss", linewidth=1.2)
    ax.plot(val_epochs, val_losses, label="Val loss", linewidth=1.5, marker="o", markersize=3)
    if best_epoch is not None:
        ax.axvline(best_epoch, color="red", linestyle="--", linewidth=1, label=f"Best (epoch {best_epoch})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    h = config.model.hidden_dim
    l = config.model.num_layers
    ax.set_title(f"Loss curve — {mode} | h{h} l{l}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    plot_dir = Path(config.training.checkpoint_dir).parent
    plot_path = plot_dir / f"loss_curve_h{h}_l{l}_{mode}.png"
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)
    print(f"Loss curve saved to: {plot_path}")

    if use_wandb:
        wandb.log({"loss_curve": wandb.Image(str(plot_path))})
        wandb.finish()
    # ─────────────────────────────────────────────────────────────────────────

    print("\n" + "=" * 80)
    print(f"Training complete! ({mode} mode)")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {config.training.checkpoint_dir}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train diffusion model in raw or image mode")
    parser.add_argument(
        "--mode",
        choices=["raw", "image", "decomposition"],
        default="raw",
        help="Training mode: 'raw', 'image', or 'decomposition' (raw + trend/FFT loss)"
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
    parser.add_argument(
        "--fft-weight",
        type=float,
        default=None,
        help="FFT auxiliary loss weight (0=off). decomposition mode default: 0.1"
    )
    parser.add_argument(
        "--trend-weight",
        type=float,
        default=None,
        help="Trend loss weight via moving-average decomposition (0=off). decomposition mode default: 0.5"
    )
    parser.add_argument(
        "--season-weight",
        type=float,
        default=None,
        help="Seasonal residual loss weight (0=off). decomposition mode default: 0.0"
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
        fft_weight=args.fft_weight,
        trend_weight=args.trend_weight,
        season_weight=args.season_weight,
    )
