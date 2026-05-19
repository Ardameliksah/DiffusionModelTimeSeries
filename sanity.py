"""
Train from scratch for 100 epochs with inline evaluation every 10 epochs.

Every 10 epochs:
  - Generate samples
  - Compute discriminative + predictive scores
  - Log per-feature stats to sanity.txt
  - Save 4-sample real-vs-generated comparison plot to sanity_images/

Usage:
    python sanity.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config.stocks_config import Config
from utils.data_utils import create_data_loaders
from utils import create_optimizer_and_scheduler, set_seed
from models import create_model
from eval_metrics import discriminative_score, predictive_score

# ── Configuration ─────────────────────────────────────────────────────────────
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
HIDDEN_DIM   = 64
NUM_LAYERS   = 3
LR           = 1e-4
TOTAL_EPOCHS = 100
EVAL_EVERY   = 10      # evaluate disc/pred and save plot every N epochs
NUM_SAMPLES  = 256     # samples for disc/pred scoring
PLOT_SAMPLES = 4       # rows shown in comparison plots
DDIM_STEPS   = 100     # DDIM steps during training eval (fast)
FEATURE_NAMES = ["Open", "High", "Low", "Close", "Adj_Close", "Volume"]
SEED         = 42

SANITY_TXT     = Path(__file__).parent / "sanity.txt"
SANITY_IMG_DIR = Path(__file__).parent / "sanity_images"
SANITY_IMG_DIR.mkdir(exist_ok=True)

set_seed(SEED)

# ── Data ──────────────────────────────────────────────────────────────────────
config = Config()
config.model.hidden_dim  = HIDDEN_DIM
config.model.num_layers  = NUM_LAYERS
config.model.ff_dim      = HIDDEN_DIM * 4
config.training.learning_rate = LR
config.training.num_epochs    = TOTAL_EPOCHS

train_loader, test_loader, _ = create_data_loaders(
    csv_path=config.data.data_path,
    batch_size=config.training.batch_size,
    window_length=config.model.sequence_length,
    neg_one_to_one=config.data.neg_one_to_one,
    train_ratio=config.data.train_split,
    num_workers=0,
    per_window=config.data.per_window_norm,
)

# Load a fixed real batch for evaluation (same every time)
real_chw = next(iter(test_loader))
while real_chw.shape[0] < NUM_SAMPLES:
    extra = next(iter(test_loader))
    real_chw = torch.cat([real_chw, extra], dim=0)
real_chw = real_chw[:NUM_SAMPLES].cpu().numpy()     # (N, C, T)
real      = real_chw.transpose(0, 2, 1)              # (N, T, C)

print(f"Device: {DEVICE}")
print(f"Model: hidden_dim={HIDDEN_DIM}, num_layers={NUM_LAYERS}")
print(f"Real data: {real.shape}  range [{real.min():.3f}, {real.max():.3f}]")
print(f"Train batches/epoch: {len(train_loader)}")

# ── Model + optimiser ─────────────────────────────────────────────────────────
model = create_model(config, model_type="raw", device=DEVICE)
total_steps = len(train_loader) * TOTAL_EPOCHS
optimizer, scheduler = create_optimizer_and_scheduler(
    model,
    learning_rate=LR,
    weight_decay=config.training.weight_decay,
    num_epochs=TOTAL_EPOCHS,
    warmup_steps=config.training.warmup_steps,
    scheduler_type=config.training.lr_scheduler_type,
    total_steps=total_steps,
)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}\n")

# ── Helpers ───────────────────────────────────────────────────────────────────
def feat_stats(data):
    out = {}
    for i, name in enumerate(FEATURE_NAMES):
        col = data[:, :, i]
        out[name] = {
            "mean":  float(col.mean()),
            "std":   float(col.std()),
            "t_std": float(col.std(axis=1).mean()),
        }
    return out


def generate(model, n):
    model.eval()
    with torch.no_grad():
        fake_chw = model.sample(
            batch_size=n,
            sampler_type="ddim",
            num_steps=DDIM_STEPS,
            eta=0.0,
        ).cpu().numpy()
    model.train()
    return fake_chw.transpose(0, 2, 1)   # (N, T, C)


def save_plot(real, fake, epoch, out_path):
    n_cols = len(FEATURE_NAMES)
    fig, axes = plt.subplots(PLOT_SAMPLES, n_cols, figsize=(n_cols * 3, PLOT_SAMPLES * 2))
    fig.suptitle(f"Epoch {epoch} — Real (blue) vs Generated (orange)", fontsize=11)
    for row in range(PLOT_SAMPLES):
        for col, name in enumerate(FEATURE_NAMES):
            ax = axes[row, col]
            ax.plot(real[row, :, col],  color="steelblue",  lw=1.0, label="Real")
            ax.plot(fake[row, :, col],  color="darkorange", lw=1.0, label="Gen")
            if row == 0:
                ax.set_title(name, fontsize=9)
            if col == 0:
                ax.set_ylabel(f"S{row+1}", fontsize=8)
            ax.tick_params(labelsize=6)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=90)
    plt.close(fig)


def evaluate(epoch, model, real, txt_file):
    fake = generate(model, NUM_SAMPLES)

    r_stats = feat_stats(real)
    f_stats = feat_stats(fake)

    disc, acc = discriminative_score(real, fake, iterations=2000, device=DEVICE)
    mae       = predictive_score(real, fake, iterations=5000, device=DEVICE)

    gen_tstd = float(fake.std(axis=1).mean())

    print(f"  [eval]  disc={disc:.4f}  acc={acc:.4f}  pred_mae={mae:.4f}  "
          f"gen_tstd={gen_tstd:.4f}  gen_mean={fake.mean():.4f}")

    with open(txt_file, "a") as f:
        f.write(f"Epoch {epoch:>4}  disc={disc:.4f}  acc={acc:.4f}  "
                f"pred_mae={mae:.4f}  gen_tstd={gen_tstd:.4f}\n")
        f.write(f"  {'Feature':<14} {'R_mean':>8} {'R_std':>8} {'R_tstd':>8} "
                f"{'G_mean':>8} {'G_std':>8} {'G_tstd':>8}\n")
        for name in FEATURE_NAMES:
            r, g = r_stats[name], f_stats[name]
            f.write(f"  {name:<14} {r['mean']:>8.4f} {r['std']:>8.4f} {r['t_std']:>8.4f} "
                    f"{g['mean']:>8.4f} {g['std']:>8.4f} {g['t_std']:>8.4f}\n")
        f.write("\n")

    img_path = SANITY_IMG_DIR / f"epoch_{epoch:04d}.png"
    save_plot(real, fake, epoch, img_path)

# ── Initialise sanity.txt ─────────────────────────────────────────────────────
with open(SANITY_TXT, "w") as f:
    f.write(f"Sanity train-from-scratch  hidden={HIDDEN_DIM}  layers={NUM_LAYERS}  "
            f"lr={LR}  epochs={TOTAL_EPOCHS}  eval_every={EVAL_EVERY}\n")
    f.write(f"Real data: {real.shape}  range [{real.min():.4f}, {real.max():.4f}]\n\n")
    f.write("REAL DATA REFERENCE:\n")
    f.write(f"  {'Feature':<14} {'mean':>8} {'std':>8} {'t_std':>8}\n")
    for name, s in feat_stats(real).items():
        f.write(f"  {name:<14} {s['mean']:>8.4f} {s['std']:>8.4f} {s['t_std']:>8.4f}\n")
    f.write("\n")

# ── Training loop ─────────────────────────────────────────────────────────────
print("=" * 60)
print(f"Training from scratch for {TOTAL_EPOCHS} epochs")
print(f"Evaluating every {EVAL_EVERY} epochs")
print("=" * 60)

for epoch in range(1, TOTAL_EPOCHS + 1):
    model.train()
    total_loss = 0.0

    for batch in train_loader:
        batch = batch.to(DEVICE)
        optimizer.zero_grad()
        loss = model.compute_loss(batch, loss_type="l1")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    lr_now   = optimizer.param_groups[0]["lr"]
    print(f"Epoch {epoch:>3}/{TOTAL_EPOCHS}  loss={avg_loss:.4f}  lr={lr_now:.2e}", end="")

    if epoch % EVAL_EVERY == 0:
        print()
        evaluate(epoch, model, real, SANITY_TXT)
    else:
        print()

print(f"\nDone.\nResults → {SANITY_TXT}\nImages  → {SANITY_IMG_DIR}/")
