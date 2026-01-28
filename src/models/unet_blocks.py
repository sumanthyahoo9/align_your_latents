"""
Building blocks of the U-net
"""
import torch
import torch.nn as nn
from src.modules.spatial_attention import SpatialAttention

class DownBlock(nn.Module):
    """
    ONE down-block
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int = 1280,
        num_groups: int = 32,
        use_attention: bool = False
    ):
        super().__init__()
        
        # First conv block
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(num_groups, out_channels)
        
        # Time embedding projection
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        
        # Second conv block
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups, out_channels)
        
        # Residual connection (if channels change)
        self.residual_conv = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        
        # Optional attention
        self.attention = SpatialAttention(out_channels) if use_attention else nn.Identity()
        
        # Downsample
        self.downsample = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)
        
        self.silu = nn.SiLU()
    
    def forward(self, x, t_emb):
        """
        Forward pass
        """
        residual = x
        
        # First conv block
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.silu(x)
        
        # Add time embedding (broadcast over spatial dims)
        t = self.time_mlp(t_emb)
        t = t[:, :, None, None]  # (B, C, 1, 1)
        x = x + t
        
        # Second conv block
        x = self.conv2(x)
        x = self.norm2(x)
        
        # Residual connection
        residual = self.residual_conv(residual)
        x = x + residual
        x = self.silu(x)
        
        # Optional attention
        x = self.attention(x)
        
        # Downsample
        x = self.downsample(x)
        return x

class UpBlock(nn.Module):
    """
    Upsampling block for U-Net decoder
    Input: (B, in_channels, H, W) + skip (B, skip_channels, H*2, W*2)
    Output: (B, out_channels, H*2, W*2)
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        skip_channels: int,  # From encoder skip connection
        time_emb_dim: int = 1280,
        num_groups: int = 32,
        use_attention: bool = False
    ):
        super().__init__()
        
        # Upsample first
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels, 4, stride=2, padding=1)
        
        # After concat with skip: in_channels + skip_channels
        total_channels = in_channels + skip_channels
        
        # First conv block
        self.conv1 = nn.Conv2d(total_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(num_groups, out_channels)
        
        # Time embedding
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        
        # Second conv block
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups, out_channels)
        
        # Residual
        self.residual_conv = nn.Conv2d(total_channels, out_channels, 1)
        
        # Optional attention
        self.attention = SpatialAttention(out_channels) if use_attention else nn.Identity()
        
        self.silu = nn.SiLU()
    
    def forward(
        self, 
        x: torch.Tensor, 
        skip: torch.Tensor, 
        t_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        x: (B, in_channels, H, W)
        skip: (B, skip_channels, H*2, W*2) from encoder
        t_emb: (B, time_emb_dim)
        """
        # Upsample
        x = self.upsample(x)  # (B, in_channels, H*2, W*2)
        
        # Concatenate with skip
        x = torch.cat([x, skip], dim=1)  # (B, in_channels+skip_channels, H*2, W*2)
        
        # Save for residual
        residual = x
        
        # First conv
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.silu(x)
        
        # Time embedding
        t = self.time_mlp(t_emb)[:, :, None, None]
        x = x + t
        
        # Second conv
        x = self.conv2(x)
        x = self.norm2(x)
        
        # Residual
        residual = self.residual_conv(residual)
        x = x + residual
        x = self.silu(x)
        
        # Attention
        x = self.attention(x)
        
        return x

class ResBlock(nn.Module):
    """
    ONE Res-block
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int = 1280,
        num_groups: int = 32,
        use_attention: bool = False
    ):
        super().__init__()
        
        # First conv block
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(num_groups, out_channels)
        
        # Time embedding projection
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        
        # Second conv block
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups, out_channels)
        
        # Residual connection (if channels change)
        self.residual_conv = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        
        # Optional attention
        self.attention = SpatialAttention(out_channels) if use_attention else nn.Identity()
        
        self.silu = nn.SiLU()
    
    def forward(self, x, t_emb):
        """
        Forward pass
        """
        residual = x
        
        # First conv block
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.silu(x)
        
        # Add time embedding (broadcast over spatial dims)
        t = self.time_mlp(t_emb)
        t = t[:, :, None, None]  # (B, C, 1, 1)
        x = x + t
        
        # Second conv block
        x = self.conv2(x)
        x = self.norm2(x)
        
        # Residual connection
        residual = self.residual_conv(residual)
        x = x + residual
        x = self.silu(x)
        
        # Optional attention
        x = self.attention(x)
        
        return x