"""
Test the up-convolution block
"""
import torch
from src.models.unet_blocks import UpBlock

def test_up_block():
    """
    The unit test
    """
    B, in_ch, out_ch, skip_ch = 2, 64, 128, 32
    H, W = 16, 16
    
    x = torch.randn(B, in_ch, H, W)
    skip = torch.randn(B, skip_ch, H, W)  # ← SAME size as x!
    t_emb = torch.randn(B, 1280)
    
    up_block = UpBlock(in_ch, out_ch, skip_ch)
    out = up_block(x, skip, t_emb)
    
    assert out.shape == (B, out_ch, H*2, W*2)  # ← Upsampled!