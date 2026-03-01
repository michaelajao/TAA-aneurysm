"""
Base neural network architectures for TAA-PINN

Implements networks for velocity components (u, v, w), pressure (p),
and turbulent viscosity (nut) with Fourier feature encoding and
residual blocks.
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
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
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
        if self.fourier is not None:
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


class Net2_nut(TAANet):
    """Network for non-dimensional turbulent viscosity nu_t_bar.

    Output is passed through softplus + a hard minimum floor to guarantee
    nu_t >= nu_t_min at every point, eliminating the degenerate nu_t = 0
    equilibrium that the physics loss can otherwise drive the network toward.

    Key design choice: the raw network output is SHIFTED by +2 before
    softplus so the operating point is sigmoid(+2) ≈ 0.88 (gradient),
    not sigmoid(−7) ≈ 0.001.  This prevents the vanishing-gradient
    trap.  The additive floor then ensures a non-zero baseline.

    Output = softplus(raw + SOFTPLUS_SHIFT) + nu_t_min
    Initial output ≈ softplus(bias + 2) + nu_t_min ≈ initial_nut + nu_t_min.
    """
    SOFTPLUS_SHIFT = 2.0

    def __init__(self, *args, initial_nut: float = 0.05,
                 nu_t_min: float = 0.001, **kwargs):
        super().__init__(*args, **kwargs)
        import math
        self.nu_t_min = nu_t_min
        target_raw = math.log(math.expm1(max(initial_nut, 1e-8)))
        bias_val = target_raw - self.SOFTPLUS_SHIFT
        nn.init.constant_(self.decoder.bias, bias_val)

    def forward(self, x):
        raw = super().forward(x)
        return torch.nn.functional.softplus(raw + self.SOFTPLUS_SHIFT) + self.nu_t_min


def create_taa_networks(input_dim: int = 4,
                       hidden_dim: int = 256,
                       num_layers: int = 10,
                       num_frequencies: int = 32,
                       fourier_scale: float = 10.0,
                       use_fourier: bool = True,
                       nut_hidden_dim: int = 64,
                       nut_num_layers: int = 4,
                       nu_t_min: float = 0.001,
                       device: str = 'cuda') -> dict:
    """
    Create all five networks (u, v, w, p, nut) for TAA-PINN.

    The turbulent viscosity network (nut) uses a smaller architecture
    because nu_t is a smoother field than velocity or pressure.

    Args:
        input_dim: Input dimension
        hidden_dim: Hidden dimension for u/v/w/p
        num_layers: Number of residual blocks for u/v/w/p
        num_frequencies: Fourier frequencies
        fourier_scale: Fourier scale
        use_fourier: Use Fourier encoding
        nut_hidden_dim: Hidden dimension for nut network
        nut_num_layers: Number of residual blocks for nut network
        nu_t_min: Hard minimum floor for nu_t output (prevents degenerate zero solution)
        device: Device for networks

    Returns:
        Dictionary with keys 'u', 'v', 'w', 'p', 'nut' containing networks
    """
    networks = {
        'u': Net2_u(input_dim, hidden_dim, num_layers, num_frequencies, fourier_scale, use_fourier).to(device),
        'v': Net2_v(input_dim, hidden_dim, num_layers, num_frequencies, fourier_scale, use_fourier).to(device),
        'w': Net2_w(input_dim, hidden_dim, num_layers, num_frequencies, fourier_scale, use_fourier).to(device),
        'p': Net2_p(input_dim, hidden_dim, num_layers, num_frequencies, fourier_scale, use_fourier).to(device),
        'nut': Net2_nut(input_dim, nut_hidden_dim, nut_num_layers, num_frequencies, fourier_scale, use_fourier,
                        nu_t_min=nu_t_min).to(device),
    }

    return networks


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
