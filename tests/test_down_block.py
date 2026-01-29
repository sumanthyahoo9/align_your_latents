"""
Test ONE DownBlock of the U-net
"""
import torch
from src.models.unet_blocks import DownBlock

def test_down_block():
    """
    The unit test
    """
    B, C, H, W = 2, 16, 32, 32
    x = torch.randn(B, C, H, W)
    t_emb_dim = 1280
    down_block = DownBlock(in_channels=16,
                           out_channels=32)
    t_emb = torch.randn(B, t_emb_dim)
    out = down_block(x, t_emb)
    assert out.shape == (B, C*2, H//2, W//2)

