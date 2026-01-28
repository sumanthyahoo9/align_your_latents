"""
3D convolution
"""
import torch
import torch.nn as nn

class Conv3DBlock(nn.Module):
    """
    3D Convolution block
    Input: (B, C, T, H, W)
    Output: (B, C, T, H, W)
    """
    def __init__(self, channels: int, kernel_size:int = 3):
        super().__init__()
        self.padding = (1, 1, 1)
        self.conv_3d = nn.Conv3d(channels, channels, kernel_size=kernel_size, padding=self.padding)
        self.relu = nn.ReLU()
        self.group_norm = nn.GroupNorm(num_groups=8, num_channels=channels)
    
    def forward(self, x):
        """
        Forward pass
        """
        x = self.conv_3d(x)
        x = self.group_norm(self.relu(x))
        return x
