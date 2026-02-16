"""
Boundary Condition Loss Functions

Enforces boundary conditions at wall surfaces.
"""

import torch
import torch.nn as nn


def compute_noslip_loss(net_u, net_v, net_w,
                        x_wall, y_wall, z_wall, t_phase):
    """
    Compute no-slip boundary condition loss.

    At the wall: u = v = w = 0

    Args:
        net_u, net_v, net_w: Neural networks for velocity components
        x_wall, y_wall, z_wall: Wall boundary coordinates (N, 1)
        t_phase: Cardiac phase (N, 1)

    Returns:
        loss: MSE loss for no-slip condition
    """
    # Forward pass through networks
    net_in = torch.cat((x_wall, y_wall, z_wall, t_phase), dim=1)

    u_wall = net_u(net_in).view(-1, 1)
    v_wall = net_v(net_in).view(-1, 1)
    w_wall = net_w(net_in).view(-1, 1)

    # Target: zero velocity at wall
    loss_fn = nn.MSELoss()
    loss_u = loss_fn(u_wall, torch.zeros_like(u_wall))
    loss_v = loss_fn(v_wall, torch.zeros_like(v_wall))
    loss_w = loss_fn(w_wall, torch.zeros_like(w_wall))

    # Total no-slip loss
    loss = loss_u + loss_v + loss_w

    return loss


def compute_pressure_loss(net_p,
                         x_wall, y_wall, z_wall, t_phase,
                         p_true):
    """
    Compute pressure matching loss at the wall.

    Match predicted pressure with CFD pressure data at wall.

    Args:
        net_p: Neural network for pressure
        x_wall, y_wall, z_wall: Wall coordinates (N, 1)
        t_phase: Cardiac phase (N, 1)
        p_true: True pressure from CFD (N, 1)

    Returns:
        loss: MSE loss for pressure matching
    """
    # Forward pass
    net_in = torch.cat((x_wall, y_wall, z_wall, t_phase), dim=1)
    p_pred = net_p(net_in).view(-1, 1)

    # MSE loss
    loss_fn = nn.MSELoss()
    loss = loss_fn(p_pred, p_true)

    return loss


def compute_velocity_magnitude_constraint(net_u, net_v, net_w,
                                          x, y, z, t_phase,
                                          max_velocity=3.0):
    """
    Optional: Constrain non-dimensional velocity magnitude to reasonable values.

    This can help with training stability.  max_velocity is in non-dimensional
    units (multiples of U_ref), so 3.0 means 3x the characteristic velocity.

    Args:
        net_u, net_v, net_w: Neural networks for non-dim velocity
        x, y, z: Non-dimensional coordinates (N, 1)
        t_phase: Cardiac phase (N, 1)
        max_velocity: Maximum allowed non-dim velocity magnitude

    Returns:
        loss: Penalty for velocities exceeding max_velocity
    """
    # Forward pass
    net_in = torch.cat((x, y, z, t_phase), dim=1)
    u = net_u(net_in).view(-1, 1)
    v = net_v(net_in).view(-1, 1)
    w = net_w(net_in).view(-1, 1)

    # Velocity magnitude
    vel_mag = torch.sqrt(u**2 + v**2 + w**2 + 1e-12)

    # Penalty for exceeding max_velocity
    # Use ReLU to only penalize violations
    violation = torch.relu(vel_mag - max_velocity)

    loss = violation.mean()

    return loss
