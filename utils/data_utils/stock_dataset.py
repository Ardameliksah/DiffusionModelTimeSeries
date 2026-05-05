"""
Stock time series dataset.

Responsibilities (single-responsibility):
  - Load CSV once
  - Normalize via an injected BaseScaler
  - Create sliding windows
  - Temporal train/test split

The class does NOT decide which scaler to use — that is the caller's job
(see `loader.py` or `__init__.py` helpers).
"""

import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset

from .scalers import BaseScaler


class StockDataset(Dataset):
    """
    PyTorch Dataset for windowed stock time series.

    Data layout returned by __getitem__:
        tensor of shape (num_features, window_length)   — channels-first.
    """

    def __init__(
        self,
        csv_path: str,
        scaler: BaseScaler,
        window_length: int = 32,
        stride: int = 1,
        train_ratio: float = 0.8,
    ):
        super().__init__()
        self.scaler = scaler
        self.window_length = window_length

        # --- load CSV once ---------------------------------------------------
        import pandas as pd

        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        for enc in ("utf-8", "latin-1", "iso-8859-1"):
            try:
                df = pd.read_csv(csv_path, encoding=enc, on_bad_lines="skip", engine="python")
                break
            except Exception:
                df = None
        if df is None:
            raise ValueError(f"Could not read CSV: {csv_path}")

        raw = df.values.astype(np.float32)
        self.num_features = raw.shape[1]

        # --- fit scaler and transform ----------------------------------------
        normalized = scaler.fit_transform(raw)  # shape (T, F)

        # --- sliding windows -------------------------------------------------
        windows = []
        for i in range(0, len(normalized) - window_length + 1, stride):
            w = normalized[i: i + window_length]        # (W, F)
            windows.append(w.T)                          # (F, W)  channels-first
        windows = np.array(windows, dtype=np.float32)

        # --- temporal split ---------------------------------------------------
        split = int(len(windows) * train_ratio)
        self.train_windows = torch.from_numpy(windows[:split])
        self.test_windows = torch.from_numpy(windows[split:])

        print(f"StockDataset: {len(windows)} windows  "
              f"(train={len(self.train_windows)}, test={len(self.test_windows)})")

    # ------ public helpers ---------------------------------------------------

    def get_train_data(self) -> torch.Tensor:
        return self.train_windows

    def get_test_data(self) -> torch.Tensor:
        return self.test_windows

    def denormalize(self, data) -> np.ndarray:
        """
        Reverse-transform an array of shape (B, F, W) back to original scale.
        """
        if not isinstance(data, np.ndarray):
            data = data.cpu().numpy()
        B, F, W = data.shape
        flat = data.transpose(0, 2, 1).reshape(-1, F)       # (B*W, F)
        inv = self.scaler.inverse_transform(flat)            # (B*W, F)
        return inv.reshape(B, W, F).transpose(0, 2, 1).astype(np.float32)

    # ------ Dataset interface (training windows by default) ------------------

    def __len__(self) -> int:
        return len(self.train_windows)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.train_windows[idx]
