"""
Diagnostic: x_0 range explosion in pred_noise vs pred_x0.

Loads an existing pred_noise checkpoint and shows how predict_start_from_noise
produces values far outside [-1, 1] at high noise levels due to coefficient
amplification. Also shows the raw transformer output is bounded in contrast.

Run from TezBaselines/:
    python MyCode/diagnose_x0_range.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import torch
from config.stocks_config import Config
from models.diffusion_model import DiffusionModel
from utils.data_utils.stock_dataset import StockDataset
from utils.data_utils.scalers import MinMaxNormScaler

# ---------------------------------------------------------------------------
CHECKPOINT = "MyCode/output/checkpoints_h128_l6/best_model.pt"
T_LEVELS   = [0, 100, 300, 500, 700, 900, 999]
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
BATCH      = 64
# ---------------------------------------------------------------------------


def load_model(checkpoint_path: str, device: str) -> DiffusionModel:
    ckpt = torch.load(checkpoint_path, map_location=device)
    saved_cfg = ckpt.get("model_config", {})

    cfg = Config()
    # override arch from saved config if present
    for key, val in saved_cfg.items():
        if hasattr(cfg.model, key):
            setattr(cfg.model, key, val)

    model = DiffusionModel(
        input_channels=cfg.model.input_channels,
        sequence_length=cfg.model.sequence_length,
        hidden_dim=cfg.model.hidden_dim,
        num_layers=cfg.model.num_layers,
        num_heads=cfg.model.num_heads,
        ff_dim=cfg.model.ff_dim,
        dropout=cfg.model.dropout,
        learnable_pos_enc=cfg.model.learnable_pos_enc,
        num_timesteps=cfg.diffusion.num_timesteps,
        noise_schedule=cfg.diffusion.noise_schedule,
        device=device,
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded: {checkpoint_path}")
    print(f"  hidden_dim={cfg.model.hidden_dim}, num_layers={cfg.model.num_layers}, "
          f"schedule={cfg.diffusion.noise_schedule}, T={cfg.diffusion.num_timesteps}")
    return model


def load_data(device: str) -> torch.Tensor:
    cfg     = Config()
    scaler  = MinMaxNormScaler(neg_one_to_one=True)
    dataset = StockDataset(
        csv_path=cfg.data.data_path,
        scaler=scaler,
        window_length=cfg.model.sequence_length,
        train_ratio=cfg.data.train_split,
    )
    x_0 = dataset.train_windows[:BATCH].to(device)
    print(f"Real data: shape={tuple(x_0.shape)}, "
          f"min={x_0.min():.3f}, max={x_0.max():.3f}  (should be in [-1, 1])\n")
    return x_0


def row(label: str, x: torch.Tensor) -> str:
    frac = ((x < -1) | (x > 1)).float().mean().item()
    return (f"  {label} | "
            f"min={x.min():8.3f}  max={x.max():8.3f}  "
            f"std={x.std():6.3f}  "
            f"outside [-1,1]: {frac:6.1%}")


def main():
    model = load_model(CHECKPOINT, DEVICE)
    x_0   = load_data(DEVICE)

    diffusion = model.diffusion

    # Show the amplification coefficients so the user understands why the range explodes
    print("=== Amplification coefficients of predict_start_from_noise ===")
    print("  (1/√ᾱ_t)  and  √(1/ᾱ_t − 1)  grow as t increases")
    print(f"  {'t':>5}  {'1/√ᾱ_t':>10}  {'√(1/ᾱ_t−1)':>12}")
    for t_val in T_LEVELS:
        t   = torch.tensor([t_val], device=DEVICE)
        c1  = diffusion.extract(diffusion.sqrt_recip_alphas_cumprod,  t, (1,1,1)).item()
        c2  = diffusion.extract(diffusion.sqrt_recipm1_alphas_cumprod, t, (1,1,1)).item()
        print(f"  {t_val:>5}  {c1:>10.2f}  {c2:>12.2f}")

    print()
    print("=== OLD pred_noise path: x_start range BEFORE clipping ===")
    print("  x_start = (1/√ᾱ_t)*x_t − √(1/ᾱ_t−1)*ε_pred")
    print(f"  {'t':>5}   {'min':>8}  {'max':>8}  {'std':>6}  outside [-1,1]")
    with torch.no_grad():
        for t_val in T_LEVELS:
            t              = torch.full((x_0.shape[0],), t_val, dtype=torch.long, device=DEVICE)
            x_t, _         = diffusion.q_sample(x_0, t)
            noise_pred      = model.transformer(x_t, t)          # raw model output (noise)
            x_start_old     = diffusion.predict_start_from_noise(x_t, t, noise_pred)
            print(row(f"t={t_val:4d}", x_start_old))

    print()
    print("=== Raw transformer output range (NOT passed through predict_start_from_noise) ===")
    print("  This is what the model directly produces (trained as noise predictor)")
    print(f"  {'t':>5}   {'min':>8}  {'max':>8}  {'std':>6}  outside [-1,1]")
    with torch.no_grad():
        for t_val in T_LEVELS:
            t          = torch.full((x_0.shape[0],), t_val, dtype=torch.long, device=DEVICE)
            x_t, _     = diffusion.q_sample(x_0, t)
            raw_output  = model.transformer(x_t, t)
            print(row(f"t={t_val:4d}", raw_output))

    print()
    print("=== Summary ===")
    print("OLD path: predict_start_from_noise amplifies the raw output by (1/√ᾱ_t),")
    print("          which explodes at high t → x_start goes far outside [-1, 1].")
    print("          clamp_(-1, 1) silently discards all this information during sampling.")
    print()
    print("NEW path (pred_x0): transformer output IS x_0 directly.")
    print("          After retraining, the model learns to stay near [-1, 1] at all t.")
    print("          No amplification, no information loss from clipping.")


if __name__ == "__main__":
    main()
