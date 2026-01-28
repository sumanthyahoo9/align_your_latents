"""
Reshaping utils
"""
import torch
from einops import rearrange

def spatial_to_temporal(x: torch.Tensor, t: int) -> torch.Tensor:
    """(B*T, C, H, W) -> (B, C, T, H, W)"""
    x = rearrange(x, "(b t) c h w -> b c t h w", t=t)
    return x

def temporal_to_spatial(x: torch.Tensor) -> torch.Tensor:
    """(B, C, T, H, W) -> (B*T, C, H, W)"""
    x = rearrange(x, "b c t h w -> (b t) c h w")
    return x