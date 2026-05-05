"""
Quick demo: predict_start_from_noise vs direct pred_x0 range.
No checkpoint needed — runs in seconds.
"""
import torch
import math

T      = 1000
device = "cpu"

# ── build cosine schedule (same as your GaussianDiffusion) ──────────────────
steps = torch.arange(T + 1, dtype=torch.float64)
alphas_cumprod = torch.cos(((steps / T) + 0.008) / 1.008 * math.pi / 2) ** 2
alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
alphas_cumprod = alphas_cumprod[:T].float()          # shape (T,)

sqrt_recip_ac   = (1.0 / alphas_cumprod).sqrt()      # 1/√ᾱ_t
sqrt_recipm1_ac = (1.0 / alphas_cumprod - 1).sqrt()  # √(1/ᾱ_t - 1)

# ── fake x_0 ∈ [-1, 1] and a plausible noise prediction ∈ N(0,1) ──────────
torch.manual_seed(0)
x0        = torch.rand(64, 6, 32) * 2 - 1           # (B, C, L) in [-1, 1]
noise_true = torch.randn_like(x0)                    # ground-truth noise

print(f"{'t':>5}  {'c1=1/sqrtA':>10}  {'c2=sqrt(1/A-1)':>14}  "
      f"{'pred_noise->x0 max':>18}  {'direct x0 max':>14}")
print("-" * 68)

for t_val in [0, 100, 300, 500, 700, 900, 999]:
    c1 = sqrt_recip_ac[t_val].item()
    c2 = sqrt_recipm1_ac[t_val].item()

    # Forward: x_t = √ᾱ * x0 + √(1-ᾱ) * ε
    ac  = alphas_cumprod[t_val]
    x_t = ac.sqrt() * x0 + (1 - ac).sqrt() * noise_true

    # ── OLD path: predict noise → recover x0 ─────────────────────────────
    # Simulate a realistic model: small prediction error (std=0.1)
    noise_pred    = noise_true + 0.1 * torch.randn_like(noise_true)
    x0_from_noise = c1 * x_t - c2 * noise_pred      # predict_start_from_noise

    # ── NEW path: predict x0 directly ────────────────────────────────────
    # Pretend the model outputs x0 perfectly
    x0_direct = x0                                   # perfect predictor

    print(f"{t_val:>5}  {c1:>9.2f}  {c2:>12.2f}  "
          f"{x0_from_noise.abs().max().item():>18.3f}  "
          f"{x0_direct.abs().max().item():>14.3f}")

print()
print("Even with a PERFECT noise predictor, predict_start_from_noise gives")
print("values far outside [-1, 1] at high t because of the c1/c2 blow-up.")
print("A direct pred_x0 model is constrained by whatever the network outputs.")
