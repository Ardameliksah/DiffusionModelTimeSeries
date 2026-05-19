"""
Input data inspection: dump one window as raw + normalized text.

Writes `input_window.txt` with the original CSV values (prices/volume)
and the normalized values (exactly what the transformer receives) for the
first 32-row window side by side.

Usage:
    python input_test.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np

from config.stocks_config import Config
from utils.data_utils import create_data_loaders

FEATURE_NAMES = ["Open", "High", "Low", "Close", "Adj_Close", "Volume"]
OUT_TXT = Path(__file__).parent / "input_window.txt"

config = Config()

# Load dataset (we need the dataset object for denormalize + raw CSV access)
_, _, dataset = create_data_loaders(
    csv_path=config.data.data_path,
    batch_size=64,
    window_length=config.model.sequence_length,
    neg_one_to_one=config.data.neg_one_to_one,
    train_ratio=config.data.train_split,
    num_workers=0,
    per_window=config.data.per_window_norm,
)

# Normalized window 0: shape (6, 32) → transpose to (32, 6)
norm_window = dataset.train_windows[0].numpy()   # (6, 32)
norm_T = norm_window.T                            # (32, 6)

# Raw window: denormalize window 0 back to original CSV scale
raw_chw = dataset.denormalize(norm_window[np.newaxis], window_indices=[0])[0]  # (6, 32)
raw_T = raw_chw.T  # (32, 6)

col_w = 14   # column width

lines = []
lines.append("=" * 100)
lines.append(f"Window 0  (rows 0–{config.model.sequence_length - 1} of CSV)")
lines.append("=" * 100)

# Header
hdr = f"{'Step':>4}  " + "".join(f"{f + '_raw':>{col_w}}" for f in FEATURE_NAMES)
hdr += "    " + "".join(f"{f + '_norm':>{col_w}}" for f in FEATURE_NAMES)
lines.append(hdr)
lines.append("-" * len(hdr))

for t in range(config.model.sequence_length):
    raw_vals  = "".join(f"{raw_T[t, i]:>{col_w}.4f}"  for i in range(len(FEATURE_NAMES)))
    norm_vals = "".join(f"{norm_T[t, i]:>{col_w}.6f}" for i in range(len(FEATURE_NAMES)))
    lines.append(f"{t:>4}  {raw_vals}    {norm_vals}")

lines.append("")
lines.append("SUMMARY")
lines.append("-" * 60)
lines.append(f"{'Feature':<14} {'raw_min':>12} {'raw_max':>12} {'norm_min':>10} {'norm_max':>10}")
for i, name in enumerate(FEATURE_NAMES):
    lines.append(
        f"{name:<14} {raw_T[:, i].min():>12.4f} {raw_T[:, i].max():>12.4f}"
        f" {norm_T[:, i].min():>10.6f} {norm_T[:, i].max():>10.6f}"
    )

text = "\n".join(lines)
print(text)
OUT_TXT.write_text(text, encoding="utf-8")
print(f"\nSaved: {OUT_TXT}")
