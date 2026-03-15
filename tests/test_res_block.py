"""
Test the residual block
"""
import torch
from src.models.unet_blocks import ResBlock

def test_res_block():
    """
    Unit test
    """
    B, C, H, W = 2, 16, 32, 32
    x = torch.randn(B, C, H, W)
    t_emb_dim = 1280
    res_block = ResBlock(in_channels=16,
                         out_channels=32)
    t_emb = torch.randn(B, t_emb_dim)
    out = res_block(x, t_emb)
    assert out.shape == (B, C*2, H, W)