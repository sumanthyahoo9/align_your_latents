"""
Noise scheduler
"""
import torch

class DDPMScheduler:
    """
    DDPM Noise scheduler
    """
    def __init__(
            self,
            num_timesteps: int = 1000,
            beta_start: float = 0.0001,
            beta_end: float = 0.02,
            device: str = "cpu"
    ):
        assert num_timesteps > 1, "num_timesteps must be > 1"
        assert beta_end > beta_start, "beta_end must be > beta_start"
        self.num_timesteps = num_timesteps
        self.device = device if torch.device("cuda") else "cpu"
        # Linear schedule
        self.betas = torch.linspace(
            beta_start,
            beta_end,
            num_timesteps,
            device=device,
            dtype=torch.float32
        )
        # Pre-compute useful values
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.ones(1, device=device), self.alphas_cumprod[:-1]])
        # For posterior
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
    
    def add_noise(
            self,
            x_0: torch.Tensor,
            noise: torch.Tensor,
            timesteps: torch.Tensor
    ) -> torch.Tensor:
        """
        Add Noise
        Forward diffusion: q(x_t | x_0) = √(ᾱ_t)·x_0 + √(1-ᾱ_t)·ε
        
        Args:
            x_0: Clean latents (B, C, T, H, W)
            noise: Random noise ε ~ N(0, I), same shape as x_0
            timesteps: Timestep for each sample (B,)
        """
        # Get the coefficients for this timesteps
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[timesteps]
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[timesteps]

        # Reshape for broadcasting
        sqrt_alpha_prod = sqrt_alpha_prod.view(-1, 1, 1, 1, 1)
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.view(-1, 1, 1, 1, 1)

        # Apply noise: x_t = √(ᾱ_t)·x_0 + √(1-ᾱ_t)·ε
        x_t = sqrt_alpha_prod * x_0 + sqrt_one_minus_alpha_prod * noise
        return x_t
    
    def sample_timesteps(self, batch_size: int) -> torch.Tensor:
        """
        Sample random timesteps for training
        Args:
            batch_size: Number of timesteps to sample
        
        Returns:
            timesteps: Random timesteps (B,) in range [0, num_timesteps)
        """
        return torch.randint(0, self.num_timesteps, (batch_size,), device=self.device)
