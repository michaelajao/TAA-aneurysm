"""
Boundary Condition Loss Functions

Enforces boundary conditions:
  - No-slip at wall surfaces (u = v = w = 0)
  - Pressure matching at wall surfaces
  - Inlet velocity profile (from CFD setup)
  - Outlet zero-pressure (from CFD setup)
"""

import torch
import torch.nn as nn
import numpy as np


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


# ── Inlet / Outlet utilities ─────────────────────────────────────────────

def detect_inlet_outlet(x_wall, y_wall, z_wall, tol_frac=0.01):
    """Detect inlet and outlet cross-section geometry from wall surface points.

    Assumes the axial direction is the coordinate with the largest span.
    Inlet = axial minimum, outlet = axial maximum.

    Args:
        x_wall, y_wall, z_wall: wall coordinates (N,1) tensors
        tol_frac: fraction of axial span to use as tolerance for
                  selecting wall ring points at each end

    Returns:
        dict with keys: axial_dim, inlet_center, inlet_radius, inlet_axial_pos,
                        outlet_center, outlet_radius, outlet_axial_pos
    """
    coords = torch.cat([x_wall, y_wall, z_wall], dim=1).detach().cpu().numpy()

    spans = coords.max(axis=0) - coords.min(axis=0)
    axial_dim = int(np.argmax(spans))
    other_dims = [d for d in range(3) if d != axial_dim]

    tol = tol_frac * spans[axial_dim]
    axial = coords[:, axial_dim]

    result = {}
    result['axial_dim'] = axial_dim

    for label, extreme_fn in [('inlet', np.min), ('outlet', np.max)]:
        extreme = extreme_fn(axial)
        if label == 'inlet':
            mask = axial < (extreme + tol)
        else:
            mask = axial > (extreme - tol)

        ring = coords[mask]
        center = ring[:, other_dims].mean(axis=0)
        radii = np.sqrt(((ring[:, other_dims] - center) ** 2).sum(axis=1))
        radius = float(radii.mean())

        result[f'{label}_axial_pos'] = float(ring[:, axial_dim].mean())
        result[f'{label}_center'] = center.tolist()   # [c1, c2] in the two non-axial dims
        result[f'{label}_radius'] = radius

    return result


def generate_cross_section_points(axial_pos, center, radius, axial_dim,
                                  n_radial=8, n_angular=16, device='cuda'):
    """Generate a disk of sample points at a cross-section.

    Args:
        axial_pos: position along the axial axis
        center: [c1, c2] center in the non-axial plane
        radius: cross-section radius
        axial_dim: which coordinate (0,1,2) is axial
        n_radial: number of radial layers (including center)
        n_angular: number of angular samples per layer
        device: torch device

    Returns:
        x, y, z tensors (M, 1)
    """
    other_dims = [d for d in range(3) if d != axial_dim]
    pts = []

    # Center point
    pts.append([0.0, 0.0])

    for ir in range(1, n_radial + 1):
        r = radius * ir / n_radial * 0.95  # stay 5% inside the wall
        for ia in range(n_angular):
            theta = 2.0 * np.pi * ia / n_angular
            pts.append([r * np.cos(theta), r * np.sin(theta)])

    pts = np.array(pts)  # (M, 2) offsets from center
    M = pts.shape[0]

    coords = np.zeros((M, 3))
    coords[:, axial_dim] = axial_pos
    coords[:, other_dims[0]] = center[0] + pts[:, 0]
    coords[:, other_dims[1]] = center[1] + pts[:, 1]

    x = torch.tensor(coords[:, 0:1], dtype=torch.float32, device=device)
    y = torch.tensor(coords[:, 1:2], dtype=torch.float32, device=device)
    z = torch.tensor(coords[:, 2:3], dtype=torch.float32, device=device)

    return x, y, z


def compute_inlet_velocity_loss(net_u, net_v, net_w,
                                x, y, z, t_phase,
                                u_inlet_nondim, axial_dim=0):
    """Enforce inlet velocity profile: uniform axial flow.

    Args:
        net_u, net_v, net_w: velocity networks
        x, y, z: inlet cross-section coordinates (M, 1) in standardised coords
        t_phase: cardiac phase (M, 1)
        u_inlet_nondim: non-dimensional inlet velocity (V_inlet / U_ref)
        axial_dim: which velocity component is axial (0=u, 1=v, 2=w)

    Returns:
        loss: MSE loss
    """
    net_in = torch.cat((x, y, z, t_phase), dim=1)
    u = net_u(net_in).view(-1, 1)
    v = net_v(net_in).view(-1, 1)
    w = net_w(net_in).view(-1, 1)

    loss_fn = nn.MSELoss()
    nets_list = [u, v, w]
    target_axial = torch.full_like(nets_list[axial_dim], u_inlet_nondim)
    loss = loss_fn(nets_list[axial_dim], target_axial)

    # Transverse components should be zero
    for d in range(3):
        if d != axial_dim:
            loss = loss + loss_fn(nets_list[d], torch.zeros_like(nets_list[d]))

    return loss


def compute_outlet_pressure_loss(net_p, x, y, z, t_phase):
    """Enforce zero pressure at the outlet cross-section.

    Args:
        net_p: pressure network
        x, y, z: outlet coordinates (M, 1) in standardised coords
        t_phase: cardiac phase (M, 1)

    Returns:
        loss: MSE loss for p = 0
    """
    net_in = torch.cat((x, y, z, t_phase), dim=1)
    p = net_p(net_in).view(-1, 1)

    loss_fn = nn.MSELoss()
    loss = loss_fn(p, torch.zeros_like(p))

    return loss
