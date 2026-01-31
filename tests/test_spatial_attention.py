"""
Test the spatial attention module
"""
import torch
from src.modules.spatial_attention import SpatialAttention

def test_spatial_attention():
    """
    Test the spatial attention module
    """
    B, C, H, W = 16, 512, 32, 32
    x = torch.randn(B, C, H, W)
    spatial_attn = SpatialAttention(C)
    out = spatial_attn(x)
    assert out.shape == (B, C, H, W)   