"""
Residual blocks for deeper PINN architectures
"""

import torch
import torch.nn as nn


class Swish(nn.Module):
    """
    Swish activation function: f(x) = x * sigmoid(x)

    Smooth, non-monotonic activation that often performs better than ReLU
    for physics-informed neural networks.
    """

    def forward(self, x):
        return x * torch.sigmoid(x)


class ResidualBlock(nn.Module):
    """
    Residual block with skip connection for improved gradient flow.

    Architecture:
        x -> Linear -> Activation -> Linear -> (+) -> output
        |___________________________________|
                    (skip connection)

    No activation after the skip connection — this is the standard pre-activation
    ResNet style: the activation is already applied inside the residual branch.
    A second activation after the add would squash the skip path gradients.
    """

    def __init__(self,
                 dim: int,
                 activation: nn.Module = None):
        """
        Args:
            dim: Hidden dimension
            activation: Activation function (default: Swish)
        """
        super().__init__()

        if activation is None:
            activation = Swish()

        self.layer = nn.Sequential(
            nn.Linear(dim, dim),
            activation,
            nn.Linear(dim, dim)
        )

    def forward(self, x):
        """
        Args:
            x: (N, dim) input tensor

        Returns:
            output: (N, dim) output tensor
        """
        residual = self.layer(x)
        return x + residual  # Skip connection — no extra activation


