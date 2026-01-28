"""
Test the up-convolution block
"""
import torch
from src.models.unet_blocks import UpBlock

def test_up_block():
    """
    The unit test
    """
    B, C, H, W, skip = 2, 16, 32, 32, 8
    x = torch.randn(B, C, H, W)
    up_block = UpBlock(C, C*2, skip)
    time_emb_dim = 1280
    t_emb = torch.randn(B, time_emb_dim)
    skip_tensor = torch.randn(B, skip, H*2, W*2)
    out = up_block(x, skip_tensor, t_emb)
    assert out.shape == (B, C*2, H*2, W*2)