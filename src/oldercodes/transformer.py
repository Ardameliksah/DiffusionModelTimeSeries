"""
Transformer-based model for diffusion process.
Simple sequence-to-sequence transformer that predicts noise from noisy time series.
"""

import math
import torch
import torch.nn as nn
from typing import Tuple


class SinusoidalEmbedding(nn.Module):
    """Sinusoidal positional embedding for timesteps."""
    
    def __init__(self, dim: int):
        """
        Args:
            dim: Embedding dimension
        """
        super().__init__()
        self.dim = dim
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Create sinusoidal embeddings for timesteps.
        
        Args:
            t: Tensor of timestep indices, shape (batch_size,)
        
        Returns:
            Embeddings of shape (batch_size, dim)
        """
        device = t.device
        half_dim = self.dim // 2
        
        # Create frequency bands
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        
        # Compute sinusoidal embeddings
        embeddings = t.unsqueeze(1) * embeddings.unsqueeze(0)
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        
        return embeddings


class TimeEmbedding(nn.Module):
    """Project timestep embeddings to model dimension."""
    
    def __init__(self, dim: int, hidden_dim: int):
        """
        Args:
            dim: Sinusoidal embedding dimension
            hidden_dim: Output dimension (model hidden dimension)
        """
        super().__init__()
        self.embedding = SinusoidalEmbedding(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Timestep indices, shape (batch_size,)
        
        Returns:
            Embeddings of shape (batch_size, hidden_dim)
        """
        embeddings = self.embedding(t)
        return self.mlp(embeddings)


class TransformerBlock(nn.Module):
    """Single transformer encoder block with self-attention and feedforward."""
    
    def __init__(self, hidden_dim: int, num_heads: int, ff_dim: int, dropout: float = 0.1):
        """
        Args:
            hidden_dim: Model hidden dimension
            num_heads: Number of attention heads
            ff_dim: Feedforward inner dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        # Layer normalization before attention (pre-norm)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        # Layer normalization before feedforward
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_dim),
            nn.Dropout(dropout),
        )
    
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_dim)
            time_emb: Time embedding of shape (batch_size, hidden_dim)
        
        Returns:
            Output tensor of shape (batch_size, seq_len, hidden_dim)
        """
        # Self-attention with residual connection
        normalized_x = self.norm1(x)
        attn_out, _ = self.attention(normalized_x, normalized_x, normalized_x)
        x = x + attn_out
        
        # Add time embedding as modulation
        x = x + time_emb.unsqueeze(1)  # Broadcast across sequence dimension
        
        # Feedforward with residual connection
        normalized_x = self.norm2(x)
        ff_out = self.feedforward(normalized_x)
        x = x + ff_out
        
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
    Transformer-based model for diffusion process.
    
    Input: noisy time series (batch_size, channels, seq_len) + timestep
    Output: predicted noise (batch_size, channels, seq_len)
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
        """
        Args:
            input_channels: Number of input channels (6 for stock data)
            sequence_length: Length of time series (32)
            hidden_dim: Hidden dimension throughout the model
            num_layers: Number of transformer blocks
            num_heads: Number of attention heads
            ff_dim: Feedforward inner dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        self.input_channels = input_channels
        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim
        
        # Input projection: (batch, channels, seq_len) -> (batch, seq_len, hidden_dim)
        self.input_projection = nn.Linear(input_channels, hidden_dim)
        
        # Positional encoding: learnable (Diffusion-TS style) or fixed sinusoidal
        if learnable_pos_enc:
            self.pos_enc = LearnablePositionalEncoding(hidden_dim, dropout=dropout, max_len=sequence_length)
        else:
            self.pos_enc = SinusoidalPositionalEncoding(hidden_dim, dropout=dropout, max_len=sequence_length)
        
        # Time embedding
        self.time_embedding = TimeEmbedding(hidden_dim, hidden_dim)
        
        # Transformer encoder blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        
        # Output projection: (batch, seq_len, hidden_dim) -> (batch, seq_len, channels)
        self.output_projection = nn.Linear(hidden_dim, input_channels)
    
    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Predict noise from noisy time series and timestep.
        
        Args:
            x_t: Noisy time series of shape (batch_size, channels, sequence_length)
            t: Timestep indices of shape (batch_size,)
        
        Returns:
            Predicted noise of shape (batch_size, channels, sequence_length)
        """
        batch_size = x_t.shape[0]
        
        # Input projection: (batch, channels, seq_len) -> (batch, seq_len, hidden_dim)
        # Need to permute for linear layer: (batch, seq_len, channels) -> apply projection
        x = x_t.permute(0, 2, 1)  # (batch, seq_len, channels)
        x = self.input_projection(x)  # (batch, seq_len, hidden_dim)
        x = self.pos_enc(x)  # + learnable positional encoding
        
        # Time embedding: (batch,) -> (batch, hidden_dim)
        time_emb = self.time_embedding(t)
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x, time_emb)
        
        # Output projection: (batch, seq_len, hidden_dim) -> (batch, seq_len, channels)
        x = self.output_projection(x)
        
        # Permute back to channels-first format: (batch, seq_len, channels) -> (batch, channels, seq_len)
        x = x.permute(0, 2, 1)
        
        return x
    
    def get_parameter_count(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the model
    device = "cpu"
    model = TransformerDiffusionModel(
        input_channels=6,
        sequence_length=32,
        hidden_dim=128,
        num_layers=6,
        num_heads=8,
        ff_dim=512,
        dropout=0.1,
    ).to(device)
    
    # Create dummy data
    batch_size = 16
    x_t = torch.randn(batch_size, 6, 32).to(device)
    t = torch.randint(0, 1000, (batch_size,)).to(device)
    
    # Forward pass
    output = model(x_t, t)
    
    print(f"Input shape: {x_t.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Parameter count: {model.get_parameter_count():,}")
    
    assert output.shape == x_t.shape, "Output shape mismatch!"
    print("Model test passed!")
