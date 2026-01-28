"""
A temporal layer which goes in between spatial layers
"""
import torch.nn as nn
from src.modules.temporal_attention import TemporalAttention
from src.modules.conv_3d import Conv3DBlock

class TemporalLayer(nn.Module):
    """
    Create a temporal layer based on the requirements
    """
    def __init__(self, channels, layer_type: str, num_heads: int=8):
        super().__init__()
        if layer_type == "attention":
            self.temporal_module = TemporalAttention(channels=channels, num_heads=num_heads)
        elif layer_type == "conv3d":
            self.temporal_module = Conv3DBlock(channels=channels)
        else:
            raise ValueError(f"Unknown layer_type: {layer_type}. Use 'attention' or 'conv3d'")
    
    def forward(self, x):
        """
        Forward pass
        """
        out = self.temporal_module(x)
        return out