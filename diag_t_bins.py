"""
Diagnostic: per-timestep-bin analysis of model predictions.

For each of N evenly-spaced t bins, we:
  1. Take a batch of real x_0
  2. Add noise at that t  ->  x_t
  3. Ask the model to predict x_0 from x_t
  4. Report mean, std, and temporal_std of:
       - real x_0
       - noisy x_t
       - predicted x_0

Temporal std = std over the 32 time positions (within each sample), then averaged.
A flat predicted x_0 will have temporal_std ≈ 0.

Usage:
    python diag_t_bins.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import numpy as np

from config.stocks_config import Config
from utils.data_utils import create_data_loaders
from models import create_model

CHECKPOINT = Path(__file__).parent / "output" / "checkpoints_h64_l3" / "checkpoint_epoch_5000.pt"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
BATCH      = 256
N_BINS     = 10   # evenly spaced over [0, T)

# ── Load config + real data ──────────────────────────────────────────────────
config = Config()
_, test_loader, _ = create_data_loaders(
    csv_path=config.data.data_path,
    batch_size=BATCH,
    window_length=config.model.sequence_length,
    neg_one_to_one=config.data.neg_one_to_one,
    train_ratio=config.data.train_split,
    num_workers=0,
    per_window=config.data.per_window_norm,
)
real_chw = next(iter(test_loader)).to(DEVICE)   # (N, C, T)

# ── Load model ───────────────────────────────────────────────────────────────
print(f"Loading checkpoint: {CHECKPOINT}")
ckpt = torch.load(CHECKPOINT, map_location=DEVICE)
saved_cfg = ckpt.get("model_config", {})
for k, v in saved_cfg.items():
    if hasattr(config.model, k):
        setattr(config.model, k, v)

model = create_model(config, model_type="raw", device=DEVICE)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

diffusion = model.diffusion_model.diffusion   # GaussianDiffusion object
T = diffusion.num_timesteps

# ── Per-bin analysis ─────────────────────────────────────────────────────────
bin_edges = np.linspace(0, T, N_BINS + 1, dtype=int)
bin_centers = ((bin_edges[:-1] + bin_edges[1:]) / 2).astype(int)

print(f"\n{'t':>6} | {'real t_std':>10} {'x_t t_std':>10} {'pred t_std':>10}"
      f" | {'pred mean':>9} {'pred std':>9} {'real mean':>9} {'real std':>9}"
      f" | {'loss_w':>8}")
print("-" * 90)

with torch.no_grad():
    for t_val in bin_centers:
        t = torch.full((BATCH,), t_val, dtype=torch.long, device=DEVICE)

        # Forward: add noise
        x_t, _ = diffusion.q_sample(real_chw, t)

        # Predict x_0
        pred_x0 = model.diffusion_model.transformer(x_t, t)
        pred_x0 = pred_x0.clamp(-1.0, 1.0)

        # (N, C, T) -> compute temporal std: std over T dimension, mean over N and C
        def temporal_std(x):
            return x.std(dim=2).mean().item()

        r_tstd  = temporal_std(real_chw)
        xt_tstd = temporal_std(x_t)
        p_tstd  = temporal_std(pred_x0)

        p_mean = pred_x0.mean().item()
        p_std  = pred_x0.std().item()
        r_mean = real_chw.mean().item()
        r_std  = real_chw.std().item()

        lw = diffusion.loss_weight[t_val].item()

        print(f"{t_val:>6} | {r_tstd:>10.4f} {xt_tstd:>10.4f} {p_tstd:>10.4f}"
              f" | {p_mean:>9.4f} {p_std:>9.4f} {r_mean:>9.4f} {r_std:>9.4f}"
              f" | {lw:>8.4f}")

print("\nColumns:")
print("  t          — noise level (0=clean, T=pure noise)")
print("  real t_std — std over 32 timesteps in real x_0 (reference)")
print("  x_t t_std  — std over 32 timesteps in noisy x_t (should decrease as t↑)")
print("  pred t_std — std over 32 timesteps in model prediction (should match real at low t)")
print("  loss_w     — DiffusionTS loss weight at this t (high = model penalised more here)")
print()
print("Ideal: pred t_std ≈ real t_std at low t, loss_w peaks at low-to-mid t")
print("Flat-line symptom: pred t_std ≈ 0 at ALL t values")
