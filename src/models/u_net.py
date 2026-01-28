"""
The full U-net for noising and denoising
"""
import torch
import torch.nn as nn
from src.models.unet_blocks import UpBlock, DownBlock, ResBlock
from src.modules.time_embedding import TimeEmbedding
from src.modules.spatial_attention import SpatialAttention

class UNet(nn.Module):
    """
    Complete U-Net for latent diffusion
    Input: (B, 4, 64, 64) latent + timestep
    Output: (B, 4, 64, 64) predicted noise
    """
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        channels: list = [320, 640, 1280, 1280],
        time_emb_dim: int = 1280,
        num_res_blocks: int = 2,
        attention_levels: list = [1, 2, 3]
    ):
        super().__init__()
        
        # Time embedding (sinusoidal + MLP)
        self.time_embed = TimeEmbedding(time_emb_dim)
        
        # Initial conv
        self.conv_in = nn.Conv2d(in_channels, channels[0], 3, padding=1)
        
        # Encoder
        self.encoder_blocks = nn.ModuleList()
        self.encoder_skips = []  # Track channels for skip connections
        
        in_ch = channels[0]
        for i, out_ch in enumerate(channels):
            use_attn = i in attention_levels
            
            for _ in range(num_res_blocks):
                self.encoder_blocks.append(
                    DownBlock(in_ch, out_ch, time_emb_dim, use_attention=use_attn)
                )
                self.encoder_skips.append(out_ch)
                in_ch = out_ch
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            ResBlock(channels[-1], channels[-1], time_emb_dim),
            SpatialAttention(channels[-1]),
            ResBlock(channels[-1], channels[-1], time_emb_dim)
        )
        
        # Decoder
        self.decoder_blocks = nn.ModuleList()
        
        for i in reversed(range(len(channels))):
            out_ch = channels[i]
            use_attn = i in attention_levels
            
            for j in range(num_res_blocks + 1):  # +1 for matching encoder
                skip_ch = self.encoder_skips.pop()
                
                self.decoder_blocks.append(
                    UpBlock(in_ch, out_ch, skip_ch, time_emb_dim, use_attention=use_attn)
                )
                in_ch = out_ch
        
        # Output
        self.conv_out = nn.Sequential(
            nn.GroupNorm(32, channels[0]),
            nn.SiLU(),
            nn.Conv2d(channels[0], out_channels, 3, padding=1)
        )
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 4, 64, 64) latent
        t: (B,) timestep indices
        Returns: (B, 4, 64, 64) predicted noise
        """
        # Time embedding
        t_emb = self.time_embed(t)  # (B, 1280)
        
        # Initial conv
        x = self.conv_in(x)
        
        # Encoder + save skips
        skips = []
        for block in self.encoder_blocks:
            x = block(x, t_emb)
            skips.append(x)
        
        # Bottleneck
        x = self.bottleneck[0](x, t_emb)  # ResBlock
        x = self.bottleneck[1](x)         # Attention
        x = self.bottleneck[2](x, t_emb)  # ResBlock
        
        # Decoder with skips
        for block in self.decoder_blocks:
            skip = skips.pop()
            x = block(x, skip, t_emb)
        
        # Output
        x = self.conv_out(x)
        
        return x