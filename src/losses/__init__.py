from .wss import compute_wss_loss, compute_wss_metrics
from .physics import compute_physics_loss
from .boundary import compute_noslip_loss, compute_pressure_loss

__all__ = [
    "compute_wss_loss",
    "compute_wss_metrics",
    "compute_physics_loss",
    "compute_noslip_loss",
    "compute_pressure_loss",
]
