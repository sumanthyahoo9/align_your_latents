"""
Training config
"""
from typing import Tuple
from dataclasses import dataclass

@dataclass
class TrainingConfig:
    """
    Training config
    """
    # Model
    in_channels: int = 4
    out_channels: int = 4
    base_channels: int = 128
    channel_mult: Tuple[int] = (1, 2, 4, 4)
    num_res_blocks: int = 2
    attention_resolutions: Tuple[int] = (16, 8)
    input_resolution: int = 64
    temporal_layer_type: str = "attention"
    temporal_attention_heads: int = 8
    add_temporal_at_resolutions: Tuple[int] = (32, 16)

     # Diffusion
    num_timesteps: int = 1000
    beta_start: float = 0.0001
    beta_end: float = 0.02

    # Diffusion
    num_epochs: int = 100
    batch_size: int = 4
    learning_rate: float = 1e-4
    num_frames: int = 16 # T dimension

    # Optimization
    gradient_clip: float = 1.0
    use_ema: bool = True
    ema_decay: float = 0.9999

    # Checkpointing
    save_every: int = 1000 # Steps
    checkpoint_dir: str = "./checkpoints"
    log_every: int = 100 # Steps

    # Hardware
    device: str = "cuda"
    num_workers: int = 4
    mixed_precision: bool = True

    # Paths (to be filled later)
    data_dir: str = "./data"
    log_dir: str = "./logs"

