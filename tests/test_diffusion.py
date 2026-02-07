"""
Test the noise addition process
"""
# tests/test_diffusion.py
import torch
from src.training.diffusion import DDPMScheduler


def test_scheduler_initialization():
    """Test scheduler creates correct shapes and ranges"""
    scheduler = DDPMScheduler(num_timesteps=1000, device="cpu")
    
    # Check shapes
    assert scheduler.betas.shape == (1000,)
    assert scheduler.alphas.shape == (1000,)
    assert scheduler.alphas_cumprod.shape == (1000,)
    
    # Check ranges
    assert torch.all(scheduler.betas >= 0.0001)
    assert torch.all(scheduler.betas <= 0.02)
    assert torch.all(scheduler.alphas > 0)
    assert torch.all(scheduler.alphas <= 1)
    
    print("✓ Scheduler initialization test passed!")


def test_add_noise_shape():
    """Test noise addition preserves shape"""
    scheduler = DDPMScheduler(num_timesteps=1000, device="cpu")
    
    B, C, T, H, W = 2, 4, 8, 32, 32
    x_0 = torch.randn(B, C, T, H, W)
    noise = torch.randn(B, C, T, H, W)
    timesteps = torch.randint(0, 1000, (B,))
    
    x_t = scheduler.add_noise(x_0, noise, timesteps)
    
    assert x_t.shape == (B, C, T, H, W)
    print("✓ Add noise shape test passed!")


def test_add_noise_at_t0():
    """Test that t=0 gives almost clean image"""
    scheduler = DDPMScheduler(num_timesteps=1000, device="cpu")
    
    x_0 = torch.randn(1, 4, 8, 32, 32)
    noise = torch.randn(1, 4, 8, 32, 32)
    timesteps = torch.tensor([0])  # t=0
    
    x_t = scheduler.add_noise(x_0, noise, timesteps)
    
    # At t=0, should be very close to x_0
    # sqrt_alpha_cumprod[0] ≈ 0.99995, sqrt_one_minus ≈ 0.01
    assert torch.allclose(x_t, x_0, atol=0.1)
    print("✓ Add noise at t=0 test passed!")


def test_add_noise_at_t999():
    """Test that t=999 gives mostly noise"""
    scheduler = DDPMScheduler(num_timesteps=1000, device="cpu")
    
    x_0 = torch.randn(1, 4, 8, 32, 32)
    noise = torch.randn(1, 4, 8, 32, 32)
    timesteps = torch.tensor([999])  # t=999
    
    x_t = scheduler.add_noise(x_0, noise, timesteps)
    
    # At t=999, should be mostly noise
    # Check that x_t is closer to noise than to x_0
    dist_to_noise = torch.mean((x_t - noise) ** 2)
    dist_to_x0 = torch.mean((x_t - x_0) ** 2)
    
    assert dist_to_noise < dist_to_x0
    print("✓ Add noise at t=999 test passed!")


def test_sample_timesteps():
    """Test timestep sampling"""
    scheduler = DDPMScheduler(num_timesteps=1000, device="cpu")
    
    batch_size = 16
    timesteps = scheduler.sample_timesteps(batch_size)
    
    # Check shape
    assert timesteps.shape == (batch_size,)
    
    # Check range
    assert torch.all(timesteps >= 0)
    assert torch.all(timesteps < 1000)
    
    print("✓ Sample timesteps test passed!")


def test_noise_increases_with_timestep():
    """Test that noise amount increases with t"""
    scheduler = DDPMScheduler(num_timesteps=1000, device="cpu")
    
    x_0 = torch.randn(3, 4, 8, 32, 32)
    noise = torch.randn(3, 4, 8, 32, 32)
    
    # Test at different timesteps
    t_early = torch.tensor([100, 100, 100])
    t_late = torch.tensor([900, 900, 900])
    
    x_early = scheduler.add_noise(x_0, noise, t_early)
    x_late = scheduler.add_noise(x_0, noise, t_late)
    
    # Later timesteps should be noisier (farther from x_0)
    dist_early = torch.mean((x_early - x_0) ** 2)
    dist_late = torch.mean((x_late - x_0) ** 2)
    
    assert dist_late > dist_early
    print("✓ Noise increases with timestep test passed!")


if __name__ == "__main__":
    test_scheduler_initialization()
    test_add_noise_shape()
    test_add_noise_at_t0()
    test_add_noise_at_t999()
    test_sample_timesteps()
    test_noise_increases_with_timestep()
    
    print("\n🎉 All diffusion tests passed!")