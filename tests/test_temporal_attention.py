"""
Test the temporal attention module
"""
import torch
from src.modules.temporal_attention import TemporalAttention

def test_temporal_attention():
    """
    Unit test for the temporal attention
    """
    B, C, T, H, W = 2, 32, 16, 32, 32
    x = torch.randint(0, 256, (B, C, T, H, W)).float()
    temp_attention_module = TemporalAttention(C)
    x_temp = temp_attention_module(x)
    assert x_temp.shape == (B, C, T, H, W)

