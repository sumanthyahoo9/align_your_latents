"""
Test the Video U-net
"""
import torch
from src.models.video_ldm import VideoUnet

def test_video_unet():
    """
    Unit test the Video U-net
    """
    B, C, T, H, W = 2, 4, 8, 64, 64
    model = VideoUnet(
        in_channels=4,
        out_channels=4,
        base_channels=64,
        channel_mult=(1, 2),
        num_res_blocks=1,
        input_resolution=64,
        add_temporal_at_resolutions=(32,)
    )
    x = torch.randn(B, C, T, H, W)
    t = torch.randint(0, 1000, (B,))
    out = model(x, t)
    assert out.shape == (B, C, T, H, W)
    

