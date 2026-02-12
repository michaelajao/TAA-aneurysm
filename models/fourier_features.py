"""
Fourier Feature Encoding for improved high-frequency learning in PINNs
"""

import torch
import torch.nn as nn
import numpy as np


class FourierFeatures(nn.Module):
    """
    Fourier feature encoding layer.

    Maps input x to [sin(2π B x), cos(2π B x)] where B is a random matrix.
    This helps neural networks learn high-frequency functions.

    Reference: "Fourier Features Let Networks Learn High Frequency Functions
    in Low Dimensional Domains" (Tancik et al., 2020)
    """

    def __init__(self,
                 in_features: int,
                 num_frequencies: int = 32,
                 scale: float = 10.0):
        """
        Args:
            in_features: Input dimension (e.g., 4 for x,y,z,t)
            num_frequencies: Number of frequency components
            scale: Scale factor for random frequencies
        """
        super().__init__()

        self.in_features = in_features
        self.num_frequencies = num_frequencies
        self.out_features = 2 * num_frequencies

        # Random frequency matrix (fixed after initialization)
        B = torch.randn(in_features, num_frequencies) * scale
        self.register_buffer('B', B)

    def forward(self, x):
        """
        Args:
            x: (N, in_features) input tensor

        Returns:
            features: (N, 2*num_frequencies) Fourier features
        """
        # Compute x @ B
        x_proj = 2 * np.pi * x @ self.B

        # Return [sin(x_proj), cos(x_proj)]
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class GaussianFourierFeatures(nn.Module):
    """
    Gaussian Fourier features with learnable scale parameter.
    """

    def __init__(self,
                 in_features: int,
                 num_frequencies: int = 32,
                 scale: float = 1.0,
                 learnable_scale: bool = False):
        """
        Args:
            in_features: Input dimension
            num_frequencies: Number of frequency components
            scale: Initial scale factor
            learnable_scale: If True, scale is a learnable parameter
        """
        super().__init__()

        self.in_features = in_features
        self.num_frequencies = num_frequencies
        self.out_features = 2 * num_frequencies

        # Random frequency matrix from Gaussian distribution
        B = torch.randn(in_features, num_frequencies)
        self.register_buffer('B', B)

        if learnable_scale:
            self.scale = nn.Parameter(torch.tensor(scale))
        else:
            self.register_buffer('scale', torch.tensor(scale))

    def forward(self, x):
        """
        Args:
            x: (N, in_features) input tensor

        Returns:
            features: (N, 2*num_frequencies) Fourier features
        """
        # Scale the projection
        x_proj = 2 * np.pi * self.scale * (x @ self.B)

        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
