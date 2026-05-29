"""
Run this script ONCE to pre-train TS2Vec on the stock training data and cache
the weights. Subsequent Context-FID calls will load the cached weights (~1s)
instead of training from scratch (~30-60s) every time.

Usage:
    python pretrain_ts2vec.py          # uses CPU or GPU automatically
    python pretrain_ts2vec.py --force  # retrain even if weights already exist
"""

import sys, argparse, torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config.stocks_config import Config
from utils.data_utils import create_data_loaders
from utils.ts2vec.ts2vec import TS2Vec

SAVE_PATH = Path(__file__).parent / "utils" / "ts2vec_weights_stocks.pt"
SEED = 42


def main(force: bool = False):
    if SAVE_PATH.exists() and not force:
        print(f"Weights already exist at {SAVE_PATH}")
        print("Run with --force to retrain.")
        return

    config = Config()
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Collect all training windows
    print("Loading training data...")
    train_loader, _, _ = create_data_loaders(
        csv_path=config.data.data_path,
        batch_size=256,
        window_length=config.model.sequence_length,
        neg_one_to_one=config.data.neg_one_to_one,
        train_ratio=config.data.train_split,
        num_workers=0,
        per_window=config.data.per_window_norm,
    )
    all_batches = [b.cpu().numpy() for b in train_loader]
    train_np = np.concatenate(all_batches, axis=0)  # (N_train, C, L)
    train_m  = train_np.transpose(0, 2, 1)          # (N_train, L, C)
    print(f"Training data shape: {train_m.shape}  (N={train_m.shape[0]}, L={train_m.shape[1]}, C={train_m.shape[2]})")

    # Set seeds before training for reproducibility
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    model = TS2Vec(
        input_dims=train_m.shape[-1],
        device=device,
        batch_size=8,
        lr=0.001,
        output_dims=320,
        max_train_length=3000,
    )

    print(f"Training TS2Vec (seed={SEED}, output_dims=320)...")
    loss_log = model.fit(train_m, verbose=True)
    print(f"Training complete. Final loss: {loss_log[-1]:.6f}")

    model.save(str(SAVE_PATH))
    size_mb = SAVE_PATH.stat().st_size / 1e6
    print(f"\nSaved to: {SAVE_PATH}  ({size_mb:.2f} MB)")
    print("Commit this file to GitHub so Colab can load it without retraining.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Retrain even if weights already exist")
    args = parser.parse_args()
    main(force=args.force)
