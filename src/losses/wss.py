"""
Wall Shear Stress (WSS) Loss Function

This is the CRITICAL innovation for TAA-PINN.
Computes velocity gradients at the wall and matches predicted WSS with CFD ground truth.
"""

import torch
import torch.nn as nn


def compute_wss_loss(net_u, net_v, net_w,
                     x_wall, y_wall, z_wall, t_phase,
                     wss_x_true, wss_y_true, wss_z_true,
                     normals,
                     mu=0.00125,
                     X_scale=1.0,
                     Y_scale=1.0,
                     Z_scale=1.0,
                     U_scale=1.0):
    """
    Compute WSS loss by matching velocity gradients at the wall.

    Wall Shear Stress is the tangential component of the viscous stress at the wall:
        WSS = τ_tangential = (τ · n - (τ · n · n) n)

    where τ is the viscous stress tensor:
        τ_ij = μ (∂u_i/∂x_j + ∂u_j/∂x_i) for incompressible Newtonian fluid

    Args:
        net_u, net_v, net_w: Neural networks for velocity components
        x_wall, y_wall, z_wall: Wall coordinates (N, 1) tensors with requires_grad=True
        t_phase: Cardiac phase (N, 1) tensor
        wss_x_true, wss_y_true, wss_z_true: Ground truth WSS components (N, 1)
        normals: Wall normal vectors (N, 3) pointing inward
        mu: Dynamic viscosity [Pa·s] or kinematic viscosity [m²/s] depending on formulation
        X_scale, Y_scale, Z_scale: Coordinate scaling factors
        U_scale: Velocity scaling factor

    Returns:
        loss: MSE loss between predicted and true WSS
        wss_pred: Predicted WSS components (N, 3) for monitoring
    """
    # Ensure gradients are enabled
    x_wall = x_wall.clone().detach().requires_grad_(True)
    y_wall = y_wall.clone().detach().requires_grad_(True)
    z_wall = z_wall.clone().detach().requires_grad_(True)

    # Forward pass through networks
    net_in = torch.cat((x_wall, y_wall, z_wall, t_phase), dim=1)
    u = net_u(net_in)
    v = net_v(net_in)
    w = net_w(net_in)

    # Reshape outputs
    u = u.view(-1, 1)
    v = v.view(-1, 1)
    w = w.view(-1, 1)

    # Compute all velocity gradients using automatic differentiation
    # ∂u/∂x, ∂u/∂y, ∂u/∂z
    u_x = torch.autograd.grad(u, x_wall,
                              grad_outputs=torch.ones_like(u),
                              create_graph=True,
                              retain_graph=True)[0]
    u_y = torch.autograd.grad(u, y_wall,
                              grad_outputs=torch.ones_like(u),
                              create_graph=True,
                              retain_graph=True)[0]
    u_z = torch.autograd.grad(u, z_wall,
                              grad_outputs=torch.ones_like(u),
                              create_graph=True,
                              retain_graph=True)[0]

    # ∂v/∂x, ∂v/∂y, ∂v/∂z
    v_x = torch.autograd.grad(v, x_wall,
                              grad_outputs=torch.ones_like(v),
                              create_graph=True,
                              retain_graph=True)[0]
    v_y = torch.autograd.grad(v, y_wall,
                              grad_outputs=torch.ones_like(v),
                              create_graph=True,
                              retain_graph=True)[0]
    v_z = torch.autograd.grad(v, z_wall,
                              grad_outputs=torch.ones_like(v),
                              create_graph=True,
                              retain_graph=True)[0]

    # ∂w/∂x, ∂w/∂y, ∂w/∂z
    w_x = torch.autograd.grad(w, x_wall,
                              grad_outputs=torch.ones_like(w),
                              create_graph=True,
                              retain_graph=True)[0]
    w_y = torch.autograd.grad(w, y_wall,
                              grad_outputs=torch.ones_like(w),
                              create_graph=True,
                              retain_graph=True)[0]
    w_z = torch.autograd.grad(w, z_wall,
                              grad_outputs=torch.ones_like(w),
                              create_graph=True,
                              retain_graph=True)[0]

    # Account for coordinate scaling in derivatives
    # If coordinates are normalized: x_norm = x_phys / X_scale
    # Then: ∂u/∂x_phys = (∂u/∂x_norm) * (∂x_norm/∂x_phys) = (∂u/∂x_norm) / X_scale
    u_x_scaled = u_x / X_scale
    u_y_scaled = u_y / Y_scale
    u_z_scaled = u_z / Z_scale

    v_x_scaled = v_x / X_scale
    v_y_scaled = v_y / Y_scale
    v_z_scaled = v_z / Z_scale

    w_x_scaled = w_x / X_scale
    w_y_scaled = w_y / Y_scale
    w_z_scaled = w_z / Z_scale

    # Compute stress tensor components
    # For Newtonian incompressible fluid:
    # τ_ij = μ (∂u_i/∂x_j + ∂u_j/∂x_i)
    #
    # Note: This is the deviatoric stress (not including pressure)
    # The full stress is σ_ij = -p δ_ij + τ_ij, but pressure contribution
    # is normal to the wall and doesn't contribute to tangential WSS

    # Stress tensor components (symmetric)
    tau_xx = 2 * mu * u_x_scaled
    tau_yy = 2 * mu * v_y_scaled
    tau_zz = 2 * mu * w_z_scaled

    tau_xy = mu * (u_y_scaled + v_x_scaled)
    tau_xz = mu * (u_z_scaled + w_x_scaled)
    tau_yz = mu * (v_z_scaled + w_y_scaled)

    # Extract normal components (N, 3) -> (N, 1) for each component
    n_x = normals[:, 0:1]
    n_y = normals[:, 1:2]
    n_z = normals[:, 2:3]

    # Compute traction vector: t = τ · n
    # t_i = τ_ij * n_j
    t_x = tau_xx * n_x + tau_xy * n_y + tau_xz * n_z
    t_y = tau_xy * n_x + tau_yy * n_y + tau_yz * n_z
    t_z = tau_xz * n_x + tau_yz * n_y + tau_zz * n_z

    # Compute normal component of traction: t_normal = (t · n) * n
    # This is the part we need to subtract to get tangential component
    t_dot_n = t_x * n_x + t_y * n_y + t_z * n_z

    # Wall shear stress = tangential component = t - (t · n) * n
    wss_x_pred = t_x - t_dot_n * n_x
    wss_y_pred = t_y - t_dot_n * n_y
    wss_z_pred = t_z - t_dot_n * n_z

    # Compute MSE loss for each component
    loss_fn = nn.MSELoss()
    loss_x = loss_fn(wss_x_pred, wss_x_true)
    loss_y = loss_fn(wss_y_pred, wss_y_true)
    loss_z = loss_fn(wss_z_pred, wss_z_true)

    # Total WSS loss
    loss = loss_x + loss_y + loss_z

    # Stack predicted WSS for monitoring
    wss_pred = torch.cat([wss_x_pred, wss_y_pred, wss_z_pred], dim=1)

    return loss, wss_pred


def compute_wss_magnitude(wss_x, wss_y, wss_z):
    """
    Compute WSS magnitude from components.

    Args:
        wss_x, wss_y, wss_z: WSS components (N, 1)

    Returns:
        wss_mag: WSS magnitude (N, 1)
    """
    return torch.sqrt(wss_x**2 + wss_y**2 + wss_z**2 + 1e-12)  # Small epsilon for stability


def compute_wss_metrics(wss_pred, wss_true):
    """
    Compute validation metrics for WSS prediction.

    Args:
        wss_pred: Predicted WSS (N, 3) or (N, 1) for magnitude
        wss_true: True WSS (N, 3) or (N, 1)

    Returns:
        metrics: Dictionary of error metrics
    """
    # Relative L2 error
    numerator = torch.norm(wss_pred - wss_true)
    denominator = torch.norm(wss_true)
    relative_l2 = (numerator / (denominator + 1e-12)).item()

    # Pointwise relative error
    pointwise_error = torch.abs(wss_pred - wss_true) / (torch.abs(wss_true) + 1e-12)
    mean_pointwise = pointwise_error.mean().item()
    max_pointwise = pointwise_error.max().item()

    # Correlation coefficient
    wss_pred_flat = wss_pred.flatten()
    wss_true_flat = wss_true.flatten()

    # Pearson correlation
    vx = wss_pred_flat - wss_pred_flat.mean()
    vy = wss_true_flat - wss_true_flat.mean()
    correlation = (vx * vy).sum() / (torch.sqrt((vx**2).sum()) * torch.sqrt((vy**2).sum()) + 1e-12)
    correlation = correlation.item()

    # MAE
    mae = torch.abs(wss_pred - wss_true).mean().item()

    # RMSE
    rmse = torch.sqrt(((wss_pred - wss_true)**2).mean()).item()

    metrics = {
        'relative_l2': relative_l2,
        'mean_pointwise': mean_pointwise,
        'max_pointwise': max_pointwise,
        'correlation': correlation,
        'mae': mae,
        'rmse': rmse
    }

    return metrics
