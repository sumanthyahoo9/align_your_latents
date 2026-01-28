"""
Test the 3D convolution
"""
import torch
from src.modules.conv_3d import Conv3DBlock

def test_conv3d_block():
    """
    Test the 3D Convolution block
    """
    B, C, T, H, W = 2, 32, 8, 16, 16
    x = torch.randn(B, C, T, H, W)
    block = Conv3DBlock(channels=C)
    out = block(x)
    assert out.shape == (B, C, T, H, W) 