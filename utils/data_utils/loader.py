"""
DataLoader factory — builds train / test DataLoaders from config.

Keeps loader construction separate from dataset logic (SRP).
"""

import torch
from torch.utils.data import DataLoader, Dataset

from .scalers import BaseScaler, ZScoreScaler, MinMaxNormScaler
from .stock_dataset import StockDataset


def build_scaler(neg_one_to_one: bool) -> BaseScaler:
    """Return the appropriate scaler based on the config flag."""
    if neg_one_to_one:
        return MinMaxNormScaler(neg_one_to_one=True)
    return ZScoreScaler()


class _TensorDataset(Dataset):
    """Thin wrapper so we can make a DataLoader from a plain tensor."""
    def __init__(self, tensor: torch.Tensor):
        self.tensor = tensor
    def __len__(self):
        return len(self.tensor)
    def __getitem__(self, idx):
        return self.tensor[idx]


def create_data_loaders(
    csv_path: str,
    batch_size: int,
    window_length: int,
    neg_one_to_one: bool,
    train_ratio: float = 0.8,
    num_workers: int = 0,
    per_window: bool = False,
):
    """
    Build train + test DataLoaders and return them together with the dataset.

    Returns:
        (train_loader, test_loader, dataset)
    """
    scaler = build_scaler(neg_one_to_one)

    dataset = StockDataset(
        csv_path=csv_path,
        scaler=scaler,
        window_length=window_length,
        train_ratio=train_ratio,
        per_window=per_window,
    )

    train_loader = DataLoader(
        dataset,                       # __getitem__ returns train windows
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    test_ds = _TensorDataset(dataset.get_test_data())
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, test_loader, dataset
