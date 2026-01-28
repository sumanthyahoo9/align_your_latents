"""
Temporal Attention
"""
import torch
import torch.nn as nn

class TemporalAttention(nn.Module):
    """
    Module for temporal attention
    """
    def __init__(self, channels: int, num_heads: int = 8):
        """
        Temporal self-attention over T dimension
        Input: (B, C, T, H, W)
        Output: (B, C, T, H, W)
        """
        super().__init__()
        self.num_heads = num_heads
        self.channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        """