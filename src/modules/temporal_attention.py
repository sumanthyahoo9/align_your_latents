"""
Temporal Attention
"""
import torch
import torch.nn as nn
from einops import rearrange

class TemporalAttention(nn.Module):
    """
    Module for temporal attention
    """
    def __init__(self, channels: int, num_heads: int = 8):
        """
        Temporal self-attention over T dimension
        Input: (B, C, T, H, W)
        Output: (B, C, T, H, W)
        """
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            batch_first=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        x: (B, C, T, H, W)
        """
        B, C, T, H, W = x.shape
        # Reshape to (B*H*W, T, C)
        x = rearrange(x, "b c t h w -> (b h w) t c")
        # Apply attention
        x, _ = self.mha(x, x, x)
        # Reshape back to (B, C, T, H, W)
        x = rearrange(x, "(b h w) t c -> b c t h w", b=B, h=H)
        return x