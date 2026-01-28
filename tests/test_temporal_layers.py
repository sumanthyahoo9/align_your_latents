"""
Unit tests for the temporal layers
"""
import torch
from src.modules.temporal_layers import TemporalLayer

def test_temporal_layers():
    """
    Unit test
    """
    # Test for the Conv3D block
    B, C, T, H, W = 2, 32, 8, 16, 16
    x = torch.randn(B, C, T, H, W)
    conv_layer = TemporalLayer(C, "conv3d")
    out = conv_layer(x)
    assert out.shape == (B, C, T, H, W)
    temp_layer = TemporalLayer(C, "attention")
    out = temp_layer(x)
    assert out.shape == (B, C, T, H, W)
    temp_layer = TemporalLayer(C, "convolution")
    out = temp_layer(x)
    assert out.shape == (B, C, T, H, W)