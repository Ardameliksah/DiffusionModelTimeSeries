"""
Transformer-based model for diffusion process.
Sequence-to-sequence transformer with AdaLN timestep conditioning.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalEmbedding(nn.Module):
    """Sinusoidal positional embedding for timesteps."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t.unsqueeze(1) * embeddings.unsqueeze(0)
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        return embeddings


class TimeEmbedding(nn.Module):
    """Project timestep sinusoidal embedding to model dimension."""

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.embedding = SinusoidalEmbedding(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.embedding(t))


class AdaLN(nn.Module):
    """
    Adaptive Layer Norm.
    Uses the time embedding to produce per-sample scale and shift,
    replacing the fixed learned affine parameters of standard LayerNorm.
    Initialised to identity (scale=0, shift=0) so training starts stable.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.proj = nn.Linear(hidden_dim, 2 * hidden_dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        # time_emb: (batch, hidden_dim)
        scale, shift = self.proj(F.silu(time_emb)).chunk(2, dim=-1)
        # scale, shift: (batch, hidden_dim) -> broadcast over seq_len
        return self.norm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TransformerBlock(nn.Module):
    """Transformer encoder block with AdaLN timestep conditioning."""

    def __init__(self, hidden_dim: int, num_heads: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()

        self.norm1 = AdaLN(hidden_dim)
        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )

        self.norm2 = AdaLN(hidden_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        # Self-attention with AdaLN pre-norm
        normed = self.norm1(x, time_emb)
        attn_out, _ = self.attention(normed, normed, normed)
        x = x + attn_out

        # Feedforward with AdaLN pre-norm
        normed = self.norm2(x, time_emb)
        x = x + self.feedforward(normed)

        return x


class LearnablePositionalEncoding(nn.Module):
    """Learnable positional encoding (matches Diffusion-TS / PaD-TS)."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1024):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Parameter(torch.empty(1, max_len, d_model))
        nn.init.uniform_(self.pe, -0.02, 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding (original Transformer paper)."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1024):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerDiffusionModel(nn.Module):
    """
    Transformer for diffusion with AdaLN timestep conditioning.
    Input:  (batch, channels, seq_len) + timestep indices
    Output: (batch, channels, seq_len)
    """

    def __init__(
        self,
        input_channels: int = 6,
        sequence_length: int = 32,
        hidden_dim: int = 128,
        num_layers: int = 6,
        num_heads: int = 8,
        ff_dim: int = 512,
        dropout: float = 0.1,
        learnable_pos_enc: bool = True,
    ):
        super().__init__()

        self.input_channels = input_channels
        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim

        self.input_projection = nn.Linear(input_channels, hidden_dim)

        if learnable_pos_enc:
            self.pos_enc = LearnablePositionalEncoding(hidden_dim, dropout=dropout, max_len=sequence_length)
        else:
            self.pos_enc = SinusoidalPositionalEncoding(hidden_dim, dropout=dropout, max_len=sequence_length)

        self.time_embedding = TimeEmbedding(hidden_dim, hidden_dim)

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])

        self.output_projection = nn.Linear(hidden_dim, input_channels)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x = x_t.permute(0, 2, 1)          # (batch, seq_len, channels)
        x = self.input_projection(x)        # (batch, seq_len, hidden_dim)
        x = self.pos_enc(x)

        time_emb = self.time_embedding(t)   # (batch, hidden_dim)

        for block in self.transformer_blocks:
            x = block(x, time_emb)

        x = self.output_projection(x)       # (batch, seq_len, channels)
        return x.permute(0, 2, 1)           # (batch, channels, seq_len)

    def get_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    device = "cpu"
    model = TransformerDiffusionModel(
        input_channels=6, sequence_length=32, hidden_dim=128,
        num_layers=6, num_heads=8, ff_dim=512, dropout=0.1,
    ).to(device)

    x_t = torch.randn(16, 6, 32)
    t = torch.randint(0, 1000, (16,))
    out = model(x_t, t)
    assert out.shape == x_t.shape
    print(f"Output shape: {out.shape}")
    print(f"Parameter count: {model.get_parameter_count():,}")
    print("Test passed!")
