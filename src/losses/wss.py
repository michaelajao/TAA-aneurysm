"""
Wall Shear Stress (WSS) Loss Function

This is the CRITICAL innovation for TAA-PINN.
Computes velocity gradients at the wall and matches predicted WSS with CFD ground truth.

Non-dimensional formulation (viscous scaling):
    Both the predicted and target WSS are non-dimensionalized by tau_ref = mu * U_ref / L_ref.
    In this scaling, the explicit mu factor cancels:

        tau_bar_ij = d(u_bar_i)/d(x_bar_j) + d(u_bar_j)/d(x_bar_i)
        WSS_bar = tangential component of (tau_bar . n)

    This is because:
        Physical:    tau_ij = mu * (du_i/dx_j + du_j/dx_i)
        Non-dim:     tau_bar_ij = tau_ij / tau_ref
                                = mu * (U_ref/L_ref) * (du_bar_i/dx_bar_j + ...) / (mu * U_ref/L_ref)
                                = du_bar_i/dx_bar_j + du_bar_j/dx_bar_i
"""

import torch
import torch.nn as nn


def compute_wss_loss(net_u, net_v, net_w,
                     x_wall, y_wall, z_wall, t_phase,
                     wss_x_true, wss_y_true, wss_z_true,
                     normals):
    """
    Compute WSS loss by matching non-dimensional velocity gradients at the wall.

    Both predicted and target WSS are in viscous-scaled non-dimensional units
    (divided by tau_ref = mu * U_ref / L_ref), so the mu factor cancels.

    Args:
        net_u, net_v, net_w: Neural networks for non-dim velocity components
        x_wall, y_wall, z_wall: Non-dim wall coordinates (N, 1) with requires_grad=True
        t_phase: Cardiac phase (N, 1) tensor
        wss_x_true, wss_y_true, wss_z_true: Non-dim ground truth WSS components (N, 1)
        normals: Wall normal vectors (N, 3) pointing inward

    Returns:
        loss: MSE loss between predicted and true WSS
        wss_pred: Predicted WSS components (N, 3) for monitoring
    """
    # Ensure gradients are enabled
    x_wall = x_wall.clone().detach().requires_grad_(True)
    y_wall = y_wall.clone().detach().requires_grad_(True)
    z_wall = z_wall.clone().detach().requires_grad_(True)

    # Forward pass (non-dimensional outputs)
    net_in = torch.cat((x_wall, y_wall, z_wall, t_phase), dim=1)
    u = net_u(net_in).view(-1, 1)
    v = net_v(net_in).view(-1, 1)
    w = net_w(net_in).view(-1, 1)

    # Compute all velocity gradients (non-dimensional: du_bar/dx_bar)
    u_x = torch.autograd.grad(u, x_wall, grad_outputs=torch.ones_like(u),
                              create_graph=True, retain_graph=True)[0]
    u_y = torch.autograd.grad(u, y_wall, grad_outputs=torch.ones_like(u),
                              create_graph=True, retain_graph=True)[0]
    u_z = torch.autograd.grad(u, z_wall, grad_outputs=torch.ones_like(u),
                              create_graph=True, retain_graph=True)[0]

    v_x = torch.autograd.grad(v, x_wall, grad_outputs=torch.ones_like(v),
                              create_graph=True, retain_graph=True)[0]
    v_y = torch.autograd.grad(v, y_wall, grad_outputs=torch.ones_like(v),
                              create_graph=True, retain_graph=True)[0]
    v_z = torch.autograd.grad(v, z_wall, grad_outputs=torch.ones_like(v),
                              create_graph=True, retain_graph=True)[0]

    w_x = torch.autograd.grad(w, x_wall, grad_outputs=torch.ones_like(w),
                              create_graph=True, retain_graph=True)[0]
    w_y = torch.autograd.grad(w, y_wall, grad_outputs=torch.ones_like(w),
                              create_graph=True, retain_graph=True)[0]
    w_z = torch.autograd.grad(w, z_wall, grad_outputs=torch.ones_like(w),
                              create_graph=True, retain_graph=True)[0]

    # Non-dimensional stress tensor (viscous scaling -- mu cancels)
    # tau_bar_ij = du_bar_i/dx_bar_j + du_bar_j/dx_bar_i
    tau_xx = 2.0 * u_x
    tau_yy = 2.0 * v_y
    tau_zz = 2.0 * w_z

    tau_xy = u_y + v_x
    tau_xz = u_z + w_x
    tau_yz = v_z + w_y

    # Extract normal components
    n_x = normals[:, 0:1]
    n_y = normals[:, 1:2]
    n_z = normals[:, 2:3]

    # Compute traction vector: t = tau_bar . n
    t_x = tau_xx * n_x + tau_xy * n_y + tau_xz * n_z
    t_y = tau_xy * n_x + tau_yy * n_y + tau_yz * n_z
    t_z = tau_xz * n_x + tau_yz * n_y + tau_zz * n_z

    # Compute normal component of traction: t_normal = (t . n)
    t_dot_n = t_x * n_x + t_y * n_y + t_z * n_z

    # Wall shear stress = tangential component = t - (t . n) * n
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
    return torch.sqrt(wss_x**2 + wss_y**2 + wss_z**2 + 1e-12)


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
