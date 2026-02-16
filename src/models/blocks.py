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
        x -> Linear -> Activation -> Linear -> (+) -> Activation -> output
        |___________________________________|
                    (skip connection)
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

        self.activation = activation if isinstance(activation, nn.Module) else Swish()

    def forward(self, x):
        """
        Args:
            x: (N, dim) input tensor

        Returns:
            output: (N, dim) output tensor
        """
        residual = self.layer(x)
        output = x + residual  # Skip connection
        output = self.activation(output)
        return output


class ResidualBlockWithNorm(nn.Module):
    """
    Residual block with layer normalization for improved training stability.
    """

    def __init__(self,
                 dim: int,
                 activation: nn.Module = None,
                 use_norm: bool = True):
        """
        Args:
            dim: Hidden dimension
            activation: Activation function
            use_norm: Whether to use layer normalization
        """
        super().__init__()

        if activation is None:
            activation = Swish()

        layers = []
        if use_norm:
            layers.append(nn.LayerNorm(dim))
        layers.append(nn.Linear(dim, dim))
        layers.append(activation)
        if use_norm:
            layers.append(nn.LayerNorm(dim))
        layers.append(nn.Linear(dim, dim))

        self.layer = nn.Sequential(*layers)
        self.activation = activation if isinstance(activation, nn.Module) else Swish()

    def forward(self, x):
        residual = self.layer(x)
        output = x + residual
        output = self.activation(output)
        return output
