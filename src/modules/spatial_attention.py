"""
Regular spatial attention
"""
import torch
import torch.nn as nn
from einops import rearrange

class SpatialAttention(nn.Module):
    """
    Self-attention
    """
    def __init__(self, channels, num_heads:int =8):
        super().__init__()
        assert channels%num_heads == 0
        self.mha = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            batch_first=True
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        x: (B, C, H, W)
        """
        B, C, H, W = x.shape
        # Reshape to (B, H*W, C)
        x = rearrange(x, "b c h w -> b (h w) c")
        # Apply attention
        x, _ = self.mha(x, x, x)
        # Reshape back to (B, C, H, W)
        x = rearrange(x, "b (h w) c -> b c h w", h=H, w=W)
        return x