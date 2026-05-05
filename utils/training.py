"""
Training utilities for diffusion model.
Includes loss computation, optimization, and learning rate scheduling.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
from torch.utils.data import DataLoader
from typing import Dict, Tuple, Optional
import json
from pathlib import Path

_ROOT = Path(__file__).parent.parent  # MyCode/


class Trainer:
    """Training loop manager for diffusion model."""
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: Optional[object] = None,
        device: str = "cpu",
        gradient_clip_val: float = 1.0,
    ):
        """
        Args:
            model: Diffusion model to train
            optimizer: PyTorch optimizer
            scheduler: Learning rate scheduler
            device: "cpu" or "cuda"
            gradient_clip_val: Max gradient norm for clipping
        """
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.gradient_clip_val = gradient_clip_val
        self.step_count = 0
    
    def train_step(self, batch: torch.Tensor) -> float:
        """
        Single training step.
        
        Args:
            batch: Batch of clean samples (batch_size, channels, seq_len)
        
        Returns:
            Loss value
        """
        batch = batch.to(self.device)
        self.optimizer.zero_grad()
        
        # Compute loss
        loss = self.model.compute_loss(batch, loss_type="l1")
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        if self.gradient_clip_val > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.gradient_clip_val
            )
        
        # Optimization step
        self.optimizer.step()
        
        # LR scheduling
        if self.scheduler is not None:
            self.scheduler.step()
        
        self.step_count += 1
        
        return loss.item()
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training dataloader
        
        Returns:
            Average loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in train_loader:
            loss = self.train_step(batch)
            total_loss += loss
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader) -> float:
        """
        Evaluate on validation set.
        
        Args:
            val_loader: Validation dataloader
        
        Returns:
            Average loss on validation set
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        for batch in val_loader:
            batch = batch.to(self.device)
            loss = self.model.compute_loss(batch, loss_type="l1")
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        return avg_loss


def create_optimizer_and_scheduler(
    model: nn.Module,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    num_epochs: int = 500,
    warmup_steps: int = 100,
    scheduler_type: str = "cosine",
    total_steps: Optional[int] = None,
) -> Tuple[optim.Optimizer, Optional[object]]:
    """
    Create optimizer and learning rate scheduler.
    
    Args:
        model: Model to optimize
        learning_rate: Initial learning rate
        weight_decay: L2 regularization
        num_epochs: Total number of epochs
        warmup_steps: Number of warmup steps
        scheduler_type: "cosine" or "linear"
        total_steps: Total training steps (automatically computed if None)
    
    Returns:
        Tuple of (optimizer, scheduler)
    """
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    
    # Compute total steps if not provided
    if total_steps is None:
        # Assume ~50 batches per epoch (approximate)
        total_steps = num_epochs * 50
    
    # Create learning rate scheduler
    if scheduler_type == "cosine":
        # Linear warmup + cosine annealing
        scheduler = torch.optim.lr_scheduler.ChainedScheduler([
            LinearLR(
                optimizer,
                start_factor=1e-3,
                total_iters=warmup_steps,
            ),
            CosineAnnealingLR(
                optimizer,
                T_max=total_steps - warmup_steps,
                eta_min=0,
            ),
        ])
    elif scheduler_type == "linear":
        scheduler = LinearLR(optimizer, start_factor=1.0, total_iters=total_steps)
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    return optimizer, scheduler


class TrainingLogger:
    """Logs and saves training metrics."""
    
    def __init__(self, log_dir: str = str(_ROOT / "output" / "logs")):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics: Dict[str, list] = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "learning_rate": [],
        }
    
    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: Optional[float] = None,
        learning_rate: Optional[float] = None,
    ):
        """Log metrics for an epoch."""
        self.metrics["epoch"].append(epoch)
        self.metrics["train_loss"].append(train_loss)
        self.metrics["val_loss"].append(val_loss)
        self.metrics["learning_rate"].append(learning_rate)
    
    def save(self, filename: str = "training_log.json"):
        """Save metrics to JSON file."""
        filepath = self.log_dir / filename
        with open(filepath, "w") as f:
            json.dump(self.metrics, f, indent=2)
        print(f"Saved training log to {filepath}")
    
    def get_summary(self) -> Dict:
        """Get summary statistics."""
        if not self.metrics["train_loss"]:
            return {}
        
        return {
            "total_epochs": len(self.metrics["epoch"]),
            "final_train_loss": self.metrics["train_loss"][-1],
            "final_val_loss": self.metrics["val_loss"][-1] if self.metrics["val_loss"][-1] is not None else None,
            "min_train_loss": min(self.metrics["train_loss"]),
            "min_val_loss": min([l for l in self.metrics["val_loss"] if l is not None]),
        }


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    loss: float,
    checkpoint_dir: str = str(_ROOT / "output" / "checkpoints"),
    filename: Optional[str] = None,
    model_config: dict = None,
):
    """
    Save model checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        loss: Current loss
        checkpoint_dir: Directory to save to
        filename: Custom filename (default: checkpoint_epoch_{epoch}.pt)
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    if filename is None:
        filename = f"checkpoint_epoch_{epoch:04d}.pt"
    
    filepath = checkpoint_dir / filename
    
    checkpoint = {
        "epoch": epoch,
        "loss": loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "model_config": model_config,
    }
    
    torch.save(checkpoint, filepath)
    print(f"Saved checkpoint to {filepath}")
    
    return filepath


def load_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    checkpoint_path: str,
    device: str = "cpu",
) -> int:
    """
    Load model checkpoint.
    
    Args:
        model: Model to load into
        optimizer: Optimizer to load state into
        checkpoint_path: Path to checkpoint
        device: Device to load on
    
    Returns:
        Epoch number from checkpoint
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    
    print(f"Loaded checkpoint from {checkpoint_path} (epoch {epoch})")
    return epoch
