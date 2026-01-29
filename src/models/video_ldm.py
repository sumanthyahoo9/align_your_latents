"""
The Video Latent Diffusion model which combines all modules
"""
import torch
import torch.nn as nn
from einops import rearrange
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
            print(f"=== ENCODER Level {level}: out_ch={out_ch}, current_res={current_res} ===")
            for _ in range(num_res_blocks):
                print(f"  Block {block_idx}: channels={out_ch}, res={current_res}")
                # Check if this resolution needs a temporal layer
                if current_res in add_temporal_at_resolutions:
                    print(f"Block {block_idx}: Creating temporal at res={current_res}, channels={out_ch}")
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
            if level < len(channels)-1:
                current_res //= 2

        # ========== BOTTLENECK ==========
        print(f"=== BOTTLENECK: channels={channels[-1]}, current_res={current_res} ===")
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
        # Decoder processes in reverse order and upsamples AFTER each level
        for level in reversed(range(len(channels))):
            out_ch = channels[level]
            print(f"=== DECODER Level {level}: out_ch={out_ch}, current_res={current_res} ===")
            
            for block_num in range(num_res_blocks):
                print(f"  Block {block_idx}: channels={out_ch}, res={current_res}")
                
                if current_res in add_temporal_at_resolutions and level > 0:
                    print(f"Decoder block {block_idx}: Creating temporal at res={current_res}, channels={out_ch}")
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
    
            # Upsample AFTER processing all blocks at this level
            # (UpBlock upsamples at the end, so next level will be at higher res)
            if level > 0:
                print(f"  After level {level}: upsampling {current_res} -> {current_res*2}")
                current_res *= 2
            
            print(f"=== END Level {level}: current_res={current_res} ===\n")

        print(f"Created {len(self.temporal_layers)} temporal layers at resolutions {add_temporal_at_resolutions}")
        print(f"\nFINAL temporal_block_indices: {self.temporal_block_indices}")
        print(f"Total encoder blocks: {len(self.encoder_blocks)}")
        print(f"Total bottleneck blocks: {len(self.bottleneck)}")
        print(f"Total decoder blocks: {len(self.decoder_blocks)}")

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with temporal processing
        
        Args:
            x: (B, C, T, H, W) - video latents
            t: (B,) - timesteps
        
        Returns:
            (B, C, T, H, W) - processed video
        """
        
        # ========== SECTION 1: PREPARE INPUT ==========
        B, C, T, H, W = x.shape
        
        # Get time embedding and repeat for each frame
        time_emb = self.time_embed(t)  # (B, time_emb_dim)
        time_emb = time_emb.repeat_interleave(T, dim=0)  # (B*T, time_emb_dim)
        
        # Reshape to spatial format
        x = rearrange(x, "b c t h w -> (b t) c h w")
        
        # ========== SECTION 2: INITIAL CONV ==========
        x = self.conv_in(x)
        
        # ========== SECTION 3: ENCODER WITH TEMPORAL ==========
        skips = []
        temporal_layer_idx = 0
        downsample_idx = 0
        
        for block_idx, block in enumerate(self.encoder_blocks):
            print(f"Encoder block {block_idx}: x.shape={x.shape}")
            # Spatial processing
            z = block(x, time_emb)
            print(f"  After block: z.shape={z.shape}")
            
            # Temporal processing if needed
            if block_idx in self.temporal_block_indices:
                temp_layer = self.temporal_layers[temporal_layer_idx]
                alpha = self.alphas[temporal_layer_idx]
                
                # Reshape to video
                z_video = rearrange(z, "(b t) c h w -> b c t h w", b=B, t=T)
                
                # Apply temporal layer
                z_prime_video = temp_layer(z_video)
                
                # Reshape back
                z_prime = rearrange(z_prime_video, "b c t h w -> (b t) c h w")
                
                # Alpha mixing
                x = alpha * z + (1 - alpha) * z_prime
                
                temporal_layer_idx += 1
            else:
                x = z
            
            # Save skip
            skips.append(x)
            
            # Downsample if needed
            if (block_idx + 1) % self.num_res_blocks == 0 and downsample_idx < len(self.encoder_downsamples):
                x = self.encoder_downsamples[downsample_idx](x)
                downsample_idx += 1
        
        # ========== SECTION 4: BOTTLENECK WITH TEMPORAL ==========
        for block_idx, block in enumerate(self.bottleneck):
            # Spatial processing
            z = block(x, time_emb)
            # Calculate global block index
            global_block_idx = len(self.encoder_blocks) + block_idx
            print(f"Bottleneck block {block_idx} (global={global_block_idx}): x.shape={x.shape}")
            
            # Temporal processing if needed
            if global_block_idx in self.temporal_block_indices:
                print(f"  → Using temporal layer {temporal_layer_idx}")
                temp_layer = self.temporal_layers[temporal_layer_idx]
                alpha = self.alphas[temporal_layer_idx]
                
                # Reshape to video
                z_video = rearrange(z, "(b t) c h w -> b c t h w", b=B, t=T)
                
                # Apply temporal layer
                z_prime_video = temp_layer(z_video)
                
                # Reshape back
                z_prime = rearrange(z_prime_video, "b c t h w -> (b t) c h w")
                
                # Alpha mixing
                x = alpha * z + (1 - alpha) * z_prime
                
                temporal_layer_idx += 1
            else:
                x = z
        
        # ========== SECTION 5: DECODER WITH TEMPORAL ==========
        for block_idx, block in enumerate(self.decoder_blocks):
            # Get skip connection
            skip = skips.pop()
            
            # Spatial processing (UpBlock takes 3 args)
            z = block(x, skip, time_emb)
            print(f"  After block: z.shape={z.shape}")
            # Calculate global block index
            global_block_idx = len(self.encoder_blocks) + len(self.bottleneck) + block_idx
            print(f"Decoder block {block_idx} (global={global_block_idx}): x.shape={x.shape}")
            
            # Temporal processing if needed
            if global_block_idx in self.temporal_block_indices:
                print(f"  → Using temporal layer {temporal_layer_idx}")
                temp_layer = self.temporal_layers[temporal_layer_idx]
                alpha = self.alphas[temporal_layer_idx]
                
                # Reshape to video
                z_video = rearrange(z, "(b t) c h w -> b c t h w", b=B, t=T)
                
                # Apply temporal layer
                z_prime_video = temp_layer(z_video)
                
                # Reshape back
                z_prime = rearrange(z_prime_video, "b c t h w -> (b t) c h w")
                
                # Alpha mixing
                x = alpha * z + (1 - alpha) * z_prime
                
                temporal_layer_idx += 1
            else:
                x = z
        
        # ========== SECTION 6: OUTPUT ==========
        x = self.conv_out(x)
        
        # Reshape back to video format
        x = rearrange(x, "(b t) c h w -> b c t h w", b=B, t=T)
        
        return x
