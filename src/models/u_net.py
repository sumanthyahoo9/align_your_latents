# src/models/unet.py
import torch
import torch.nn as nn
from src.modules.time_embedding import TimeEmbedding
from src.models.unet_blocks import ResBlock, UpBlock


class UNet(nn.Module):
    """
    Complete U-Net for latent diffusion with attention
    Input: (B, in_channels, H, W) + timestep
    Output: (B, out_channels, H, W)
    """
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        base_channels: int = 128,
        channel_mult: tuple = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        time_emb_dim: int = 1280,
        attention_resolutions: tuple = (16, 8),
        input_resolution: int = 64
    ):
        super().__init__()
        current_res = input_resolution
        self.num_res_blocks = num_res_blocks
        channels = [base_channels * mult for mult in channel_mult]
        
        # Time embedding
        self.time_embed = TimeEmbedding(time_emb_dim)
        
        # Input convolution
        self.conv_in = nn.Conv2d(in_channels, channels[0], kernel_size=3, padding=1)
        
        # ==================== ENCODER ====================
        self.encoder_blocks = nn.ModuleList()
        self.encoder_downsamples = nn.ModuleList()
        encoder_out_channels = []  # Track what each block outputs!

        in_ch = channels[0]
        for level, out_ch in enumerate(channels):
            for _ in range(num_res_blocks):
                use_attn = current_res in attention_resolutions
                self.encoder_blocks.append(
                    ResBlock(in_ch, out_ch, time_emb_dim, use_attention=use_attn)
                )
                encoder_out_channels.append(out_ch)  # Save what this block outputs!
                in_ch = out_ch
            
            # Downsample
            if level < len(channels) - 1:
                self.encoder_downsamples.append(
                    nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1)
                )
                current_res //= 2

        # Bottleneck
        self.bottleneck = nn.ModuleList([
            ResBlock(channels[-1], channels[-1], time_emb_dim, use_attention=True),
            ResBlock(channels[-1], channels[-1], time_emb_dim, use_attention=True)
        ])
        # After encoder:
        print(f"Total skips saved: {len(encoder_out_channels)}")
        print(f"Skips: {encoder_out_channels}")
        
        # ==================== DECODER ====================
        self.decoder_blocks = nn.ModuleList()

        # Reverse the encoder channel list to match decoder order
        decoder_skip_channels = list(reversed(encoder_out_channels))
        print(f"Decoder skips available: {len(decoder_skip_channels)}")
        decoder_blocks_needed = 0
        for level in reversed(range(len(channels))):
            out_ch = channels[level]
            
            for block_idx in range(num_res_blocks):
                do_up = (level > 0)  # Don't upsample at level 0!
                use_attn = current_res in attention_resolutions
                decoder_blocks_needed += 1
                
                # Get the matching skip channel from encoder
                skip_ch = decoder_skip_channels.pop(0)  # Pop from front!
                
                self.decoder_blocks.append(
                    UpBlock(
                        in_channels=in_ch,
                        out_channels=out_ch,
                        skip_channels=skip_ch,  # ← Now correct!
                        time_emb_dim=time_emb_dim,
                        use_attention=use_attn,
                        do_upsample=do_up
                    )
                )
                in_ch = out_ch
            
            if level > 0:
                current_res *= 2
        print(f"Decoder blocks needed: {decoder_blocks_needed}")
        print(f"But we only have: {len(decoder_skip_channels)} skips!")
        
        # ==================== OUTPUT ====================
        self.conv_out = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=channels[0]),
            nn.SiLU(),
            nn.Conv2d(channels[0], out_channels, kernel_size=3, padding=1)
        )
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: (B, in_channels, H, W) - input latent
            t: (B,) - timestep indices [0, 999]
        
        Returns:
            (B, out_channels, H, W) - predicted noise
        """
        # Get time embedding
        t_emb = self.time_embed(t)  # (B, time_emb_dim)
        
        # Input projection
        x = self.conv_in(x)  # (B, base_channels, H, W)
        
        # ==================== ENCODER ====================
        skips = []
        downsample_idx = 0
        
        for i, block in enumerate(self.encoder_blocks):
            x = block(x, t_emb)
            skips.append(x)  # Save for skip connection
            
            # Downsample after every num_res_blocks
            if (i + 1) % self.num_res_blocks == 0 and downsample_idx < len(self.encoder_downsamples):
                x = self.encoder_downsamples[downsample_idx](x)
                downsample_idx += 1
        
        # ==================== BOTTLENECK ====================
        for block in self.bottleneck:
            x = block(x, t_emb)
        
        # ==================== DECODER ====================
        for block in self.decoder_blocks:
            skip = skips.pop()
            print(f"Decoder block {i}: x.shape={x.shape}, skip.shape={skip.shape}")  # DEBUG
            x = block(x, skip, t_emb)  # UpBlock handles upsampling internally!
        
        # ==================== OUTPUT ====================
        x = self.conv_out(x)
        
        return x