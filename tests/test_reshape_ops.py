"""
Unit tests for the reshaping operations
"""
import torch
from src.utils.reshape_ops import (
    spatial_to_temporal,
    temporal_to_spatial
)

def test_spatial_to_temporal():
    """
    Test the spatial to temporal function
    """
    B, T, C, H, W = 2, 16, 3, 32, 32
    x = torch.randint(0, 256, (B*T, C, H, W))
    x_temporal = spatial_to_temporal(x, T)
    assert x_temporal.shape == (B, C, T, H, W)

def test_temporal_to_spatial():
    """
    Test the temporal to spatial function
    """
    B, T, C, H, W = 2, 16, 3, 32, 32
    x = torch.randint(0, 256, (2, 3, 16, 32, 32))
    x_spatial = temporal_to_spatial(x)
    assert x_spatial.shape == (B*T, C, H, W)

def test_round_trip():
    """
    One complete round-trip test
    """
    B, T, C, H, W = 2, 8, 64, 16, 16
    x_spatial = torch.randn(B*T, C, H, W)
    
    # Transform and back
    x_temporal = spatial_to_temporal(x_spatial, T)
    x_back = temporal_to_spatial(x_temporal)
    
    # Should be identical!
    assert torch.allclose(x_spatial, x_back)
