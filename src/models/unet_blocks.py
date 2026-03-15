"""
Building blocks of the U-net
Unit test for the down block: src/tests/test_down_block.py
Unit test for the up block: src/tests/test_up_block.py
Unit test for the res block: src/tests/test_res_block.py
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
        print(f"\nThe shape of the residual is {residual.shape} and that of the input x is {x.shape}")
        
        # First conv block
        x = self.conv1(x)
        print(f"After first convolution, the shape of x is {x.shape}")
        x = self.norm1(x)
        x = self.silu(x)
        
        # Add time embedding (broadcast over spatial dims)
        t = self.time_mlp(t_emb)
        t = t[:, :, None, None]  # (B, C, 1, 1)
        print(f"The shape of the time embedding is {t.shape}")
        x = x + t
        print(f"After adding the timesteps, the shape of x is {x.shape}")
        
        # Second conv block
        x = self.conv2(x)
        print(f"After second convolution, the shape of x is {x.shape}")
        x = self.norm2(x)
        
        # Residual connection
        residual = self.residual_conv(residual)
        print(f"After applying the residual convolution to the residual layer, the shape is {residual.shape}")
        x = x + residual
        print(f"After adding the residual the shape of the input is {x.shape}")
        x = self.silu(x)
        
        # Optional attention
        x = self.attention(x)
        print(f"After attention, the shape of the input x is {x.shape}")
        
        # Downsample
        x = self.downsample(x)
        print(f"After the downsampling, the shape of x is {x.shape}")
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
        skip_channels: int,
        time_emb_dim: int = 1280,
        num_groups: int = 32,
        use_attention: bool = False,
        do_upsample: bool = True
    ):
        super().__init__()
        self.do_upsample = do_upsample
        
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
        
        # Upsample LAST - operates on out_channels!
        self.upsample = nn.ConvTranspose2d(out_channels, out_channels, 4, stride=2, padding=1)  # ← FIX!
        
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
        print(f"\nThe shape of the input is {x.shape} and that of the skip connection is {skip.shape}")
        # Concatenate with skip
        x = torch.cat([x, skip], dim=1)  # (B, in_channels+skip_channels, H*2, W*2)
        print(f"After concatenating with the skip connection, the shape of input is {x.shape}")
        
        # Save for residual
        residual = x
        
        # First conv
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.silu(x)
        print(f"After the first convolution, norm and activation, the shape of input is {x.shape}")
        
        # Time embedding
        t = self.time_mlp(t_emb)[:, :, None, None]
        print(f"The shape of the time embedding is {t_emb.shape} and that of t is {t.shape}")
        x = x + t
        print(f"After adding the time embedding passed through the MLP with input is {x.shape}")

        # Second conv
        x = self.conv2(x)
        x = self.norm2(x)
        print(f"The shape of the input after second convolution and norm is {x.shape}")
        
        # Residual
        print(f"The shape of the residual prior to convolution is {residual.shape}")
        residual = self.residual_conv(residual)
        print(f"The shape of the residual after passing it through the residual conv is {residual.shape}")
        x = x + residual
        x = self.silu(x)
        print(f"After adding the residual to the input and activation, the shape is {x.shape}")
        
        # Attention
        x = self.attention(x)
        print(f"After attention, the shape of the input is {x.shape}")
        # Upsample
        if self.do_upsample:
            x = self.upsample(x)  # (B, in_channels, H*2, W*2)
            print(f"Since upsampling is done, the shape is {x.shape}")
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
        print(f"\nThe shape of the residual is {residual.shape}")
        
        # First conv block
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.silu(x)
        print(f"After the first convolution, norm and activation, the input shape is {x.shape}")
        
        # Add time embedding (broadcast over spatial dims)
        t = self.time_mlp(t_emb)
        print(f"The shape of the input time embedding is {t_emb.shape}")
        print(f"The shape of the time embedding after MLP is {t.shape}")
        t = t[:, :, None, None]  # (B, C, 1, 1)
        print(f"The shape of time embedding after broadcasting is {t.shape}")
        x = x + t
        print(f"The shape of the input after adding the time embedding is {x.shape}")
        
        # Second conv block
        x = self.conv2(x)
        x = self.norm2(x)
        print(f"After the second convolution and norm, the input shape: {x.shape}")
        
        # Residual connection
        residual = self.residual_conv(residual)
        print(f"The shape of residual after convolution is {residual.shape}")
        x = x + residual
        print(f"After adding residual, the input shape is {x.shape}")
        x = self.silu(x)
        print(f"After activation, the shape of input is {x.shape}")
        
        # Optional attention
        x = self.attention(x)
        print(f"After attention, the shape of x is {x.shape}")
        
        return x