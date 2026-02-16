"""
Base neural network architectures for TAA-PINN

Implements networks for velocity components (u, v, w) and pressure (p)
with Fourier feature encoding and residual blocks.
"""

import torch
import torch.nn as nn

from .fourier import FourierFeatures
from .blocks import ResidualBlock, Swish


class TAANet(nn.Module):
    """
    Base network for TAA-PINN with Fourier features and residual connections.

    Architecture:
        Input -> Fourier Features -> Linear -> [Residual Blocks] -> Linear -> Output
    """

    def __init__(self,
                 input_dim: int = 4,  # x, y, z, t_phase
                 hidden_dim: int = 256,
                 num_layers: int = 10,
                 num_frequencies: int = 32,
                 fourier_scale: float = 10.0,
                 use_fourier: bool = True):
        """
        Args:
            input_dim: Input dimension (default 4 for x,y,z,t)
            hidden_dim: Hidden layer dimension
            num_layers: Number of residual blocks
            num_frequencies: Number of Fourier frequency components
            fourier_scale: Scale for Fourier features
            use_fourier: Whether to use Fourier feature encoding
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_fourier = use_fourier

        # Fourier feature encoding (optional)
        if use_fourier:
            self.fourier = FourierFeatures(input_dim, num_frequencies, fourier_scale)
            encoder_input_dim = 2 * num_frequencies
        else:
            self.fourier = None
            encoder_input_dim = input_dim

        # Encoder: map Fourier features (or raw input) to hidden dimension
        self.encoder = nn.Sequential(
            nn.Linear(encoder_input_dim, hidden_dim),
            Swish()
        )

        # Residual blocks
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim) for _ in range(num_layers)]
        )

        # Decoder: map hidden dimension to output (scalar)
        self.decoder = nn.Linear(hidden_dim, 1)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights using Kaiming normal initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='linear')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Args:
            x: (N, input_dim) input tensor

        Returns:
            output: (N, 1) output scalar field
        """
        # Fourier encoding (if enabled)
        if self.use_fourier:
            x = self.fourier(x)

        # Encoder
        x = self.encoder(x)

        # Residual blocks
        x = self.residual_blocks(x)

        # Decoder
        output = self.decoder(x)

        return output


class Net2_u(TAANet):
    """Network for x-velocity component."""
    pass


class Net2_v(TAANet):
    """Network for y-velocity component."""
    pass


class Net2_w(TAANet):
    """Network for z-velocity component."""
    pass


class Net2_p(TAANet):
    """Network for pressure field."""
    pass


def create_taa_networks(input_dim: int = 4,
                       hidden_dim: int = 256,
                       num_layers: int = 10,
                       num_frequencies: int = 32,
                       fourier_scale: float = 10.0,
                       use_fourier: bool = True,
                       device: str = 'cuda') -> dict:
    """
    Create all four networks (u, v, w, p) for TAA-PINN.

    Args:
        input_dim: Input dimension
        hidden_dim: Hidden dimension
        num_layers: Number of residual blocks
        num_frequencies: Fourier frequencies
        fourier_scale: Fourier scale
        use_fourier: Use Fourier encoding
        device: Device for networks

    Returns:
        Dictionary with keys 'u', 'v', 'w', 'p' containing networks
    """
    networks = {
        'u': Net2_u(input_dim, hidden_dim, num_layers, num_frequencies, fourier_scale, use_fourier).to(device),
        'v': Net2_v(input_dim, hidden_dim, num_layers, num_frequencies, fourier_scale, use_fourier).to(device),
        'w': Net2_w(input_dim, hidden_dim, num_layers, num_frequencies, fourier_scale, use_fourier).to(device),
        'p': Net2_p(input_dim, hidden_dim, num_layers, num_frequencies, fourier_scale, use_fourier).to(device)
    }

    return networks


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
