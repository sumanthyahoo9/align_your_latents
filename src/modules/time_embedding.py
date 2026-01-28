"""
Time embedding module
"""
import math
import torch.nn as nn
import torch

class TimeEmbedding(nn.Module):
    """
    Convert a time step into embedding
    """
    def __init__(self, dim: int = 1280):
        """
        Converts timestep t (integer) → embedding (dim,)
        """
        super().__init__()
        self.dim = dim
        
        # MLP: dim → 4*dim → dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim)
        )
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        t: (B,) timestep indices [0, 1000]
        Returns: (B, dim) embeddings
        """
        pos_enc = self.sinusoidal_embedding(t) # Shape (B, dim)
        return self.mlp(pos_enc)
    
    def sinusoidal_embedding(self, t: torch.Tensor) -> torch.Tensor:
        """
        t: (B,) → (B, dim) using sin/cos
    
        Formula:
        emb[i] = sin(t / 10000^(2i/dim)) if i even
            = cos(t / 10000^(2i/dim)) if i odd
        """
        device = t.device
        half_dim = self.dim // 2
        
        # Frequencies
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        # Outer product
        emb = t[:, None] * emb[None, :]  # (B, half_dim)
        
        # Sin and cos
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)  # (B, dim)
        
        return emb
    
