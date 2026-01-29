"""
The Video Latent Diffusion model which combines all modules
"""
import torch
import torch.nn as nn
from src.models.u_net import UNet
from src.modules.temporal_layers import TemporalLayer

class VideoUnet(UNet):
    """
    Video diffusion model
    """
    def __init__(
        self,
        # UNet params (pass through to parent)
        in_channels: int = 4,
        out_channels: int = 4,
        base_channels: int = 128,
        channel_mult: tuple = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        time_emb_dim: int = 1280,
        attention_resolutions: tuple = (16, 8),
        input_resolution: int = 64,
        # Video-specific params
        temporal_layer_type: str = "attention",
        temporal_attention_heads: int = 8,
        add_temporal_at_resolutions: tuple = (32, 16),
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels,
            channel_mult=channel_mult,
            num_res_blocks=num_res_blocks,
            time_emb_dim=time_emb_dim,
            attention_resolutions=attention_resolutions,
            input_resolution=input_resolution)
        
        # Save the video specific params
        self.temporal_layer_type = temporal_layer_type
        self.temporal_attention_heads = temporal_attention_heads
        self.add_temporal_at_resolutions = add_temporal_at_resolutions
        
        # Calculate channels at each level
        channels = [base_channels * mult for mult in channel_mult]
        # Track which blocks need temporal layers
        self.temporal_block_indices = []  # Maps block index → True/False
        self.temporal_layers = nn.ModuleList()
        self.alphas = nn.ParameterList()
        # ========== ENCODER ==========
        current_res = input_resolution
        block_idx = 0
        for level, out_ch in enumerate(channels):
            for _ in range(num_res_blocks):
                # Check if this resolution needs a temporal layer
                if current_res in add_temporal_at_resolutions:
                    self.temporal_block_indices.append(block_idx)
                    self.temporal_layers.append(
                        TemporalLayer(
                            channels=out_ch,
                            layer_type=temporal_layer_type,
                            num_heads=temporal_attention_heads
                        )
                    )
                    self.alphas.append(nn.Parameter(torch.ones(1)))
                    block_idx += 1
            # Downsampling reduces resolution
            if level < len(channels):
                current_res //= 2
        # ========== BOTTLENECK ==========
        num_bottleneck = 2 # 2 bottleneck layers
        for _ in range(num_bottleneck):
            if current_res in add_temporal_at_resolutions:
                self.temporal_block_indices.append(block_idx)
                self.temporal_layers.append(
                    TemporalLayer(
                        channels=channels[-1],
                        layer_type=temporal_layer_type,
                        num_heads=temporal_attention_heads
                    )
                )
                self.alphas.append(nn.Parameter(torch.ones(1)))
            block_idx += 1
        # ========== DECODER ==========
        for level in reversed(range(len(channels))):
            out_ch = channels[level]
            for _ in range(num_res_blocks):
                if current_res in add_temporal_at_resolutions:
                    self.temporal_block_indices.append(block_idx)
                    self.temporal_layers.append(
                        TemporalLayer(
                            channels=out_ch,
                            layer_type=temporal_layer_type,
                            num_heads=temporal_attention_heads
                        )
                    )
                    self.alphas.append(nn.Parameter(torch.ones(1)))
                block_idx += 1
            if level > 0:
                current_res *= 2
        print(f"Created {len(self.temporal_layers)} temporal layers at resolutions {add_temporal_at_resolutions}")
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, T, H, W) video latents
        t: (B,) timesteps
        Returns (B, C, T, H, W) processed video
        """
