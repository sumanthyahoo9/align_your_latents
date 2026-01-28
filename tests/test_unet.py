# tests/test_unet.py
import torch
from src.models.u_net import UNet


def test_unet_shape():
    """Test basic forward pass shape"""
    B, C, H, W = 2, 4, 64, 64
    
    model = UNet(
        in_channels=4,
        out_channels=4,
        base_channels=64,  # Changed!
        channel_mult=(1, 2, 4),  # Changed!
        num_res_blocks=1,
        attention_resolutions=(8,),  # Changed!
        input_resolution=64  # Changed!
    )
    
    x = torch.randn(B, C, H, W)
    t = torch.randint(0, 1000, (B,))
    
    out = model(x, t)
    
    assert out.shape == (B, C, H, W)


def test_unet_gradients_flow():
    """Test gradients flow through model"""
    model = UNet(
        in_channels=4,
        out_channels=4,
        base_channels=32,  # Small for speed
        channel_mult=(1, 2),  # Just 2 levels
        num_res_blocks=1,
        attention_resolutions=(),  # No attention for speed
        input_resolution=32
    )
    
    x = torch.randn(2, 4, 32, 32, requires_grad=True)
    t = torch.randint(0, 1000, (2,))
    
    out = model(x, t)
    loss = out.sum()
    loss.backward()
    
    # Check gradients exist
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()


def test_unet_different_timesteps():
    """Different timesteps should affect output"""
    model = UNet(
        in_channels=4,
        out_channels=4,
        base_channels=32,
        channel_mult=(1, 2),
        num_res_blocks=1,
        attention_resolutions=(),
        input_resolution=32
    )
    
    x = torch.randn(1, 4, 32, 32)
    t1 = torch.tensor([0])
    t2 = torch.tensor([999])
    
    out1 = model(x, t1)
    out2 = model(x, t2)
    
    # Outputs should differ!
    assert not torch.allclose(out1, out2, atol=1e-3)


def test_unet_with_attention():
    """Test U-Net with attention layers"""
    model = UNet(
        in_channels=4,
        out_channels=4,
        base_channels=32,
        channel_mult=(1, 2),
        num_res_blocks=1,
        attention_resolutions=(16,),  # Add attention!
        input_resolution=32
    )
    
    x = torch.randn(1, 4, 32, 32)
    t = torch.tensor([500])
    
    out = model(x, t)
    
    assert out.shape == x.shape