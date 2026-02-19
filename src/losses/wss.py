"""
Wall Shear Stress (WSS) Loss Function

Computes velocity gradients at the wall and matches predicted WSS with CFD
ground truth, optionally accounting for Carreau-Yasuda non-Newtonian viscosity.

Non-dimensional formulation (viscous scaling):
    tau_ref = mu_inf * U_ref / L_ref

    Newtonian case (constant mu = mu_inf):
        tau_bar_ij = du_bar_i/dx_bar_j + du_bar_j/dx_bar_i
        (mu cancels between numerator and tau_ref)

    Non-Newtonian case (Carreau-Yasuda):
        Physical:    tau_ij = mu(gamma_dot) * (du_i/dx_j + du_j/dx_i)
        Non-dim:     tau_bar_ij = (mu(gamma_dot)/mu_inf) * (du_bar_i/dx_bar_j + ...)

        The CFD WSS targets include the variable viscosity effect, so the
        PINN prediction must also multiply by mu_ratio = mu(gamma_dot)/mu_inf
        to match.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict

from .physics import carreau_yasuda_mu_ratio, compute_shear_rate_std


def compute_wss_loss(net_u, net_v, net_w,
                     x_wall, y_wall, z_wall, t_phase,
                     wss_x_true, wss_y_true, wss_z_true,
                     normals,
                     coord_scale: float = 1.0,
                     wss_std: float = 1.0,
                     non_newtonian: Optional[Dict] = None,
                     U_ref: float = 1.0,
                     L_ref: float = 1.0):
    """
    Compute WSS loss by matching velocity gradients at the wall.

    The predicted WSS is computed in tau_bar (non-dim) units from velocity
    gradients, then divided by wss_std to match the standardised targets.
    This ensures the velocity field stays at the correct physical scale
    while the loss operates in O(1) standardised units.

    Args:
        net_u, net_v, net_w: Neural networks for non-dim velocity
        x_wall, y_wall, z_wall: Standardised wall coordinates (N,1)
        t_phase:              Cardiac phase (N, 1)
        wss_x_true, etc.:    Standardised WSS targets (N, 1), i.e. tau_bar / wss_std
        normals:              Wall normal vectors (N, 3) pointing inward
        coord_scale:          Factor to convert x_std -> x_bar
        wss_std:              WSS standardisation factor (tau_bar / wss_std = target)
        non_newtonian:        Dict with Carreau-Yasuda params, or None
        U_ref:                Reference velocity (m/s)
        L_ref:                Reference length (m)

    Returns:
        loss:     MSE between predicted and true standardised WSS
        wss_pred: Predicted standardised WSS components (N, 3)
    """
    x_wall = x_wall.clone().detach().requires_grad_(True)
    y_wall = y_wall.clone().detach().requires_grad_(True)
    z_wall = z_wall.clone().detach().requires_grad_(True)

    net_in = torch.cat((x_wall, y_wall, z_wall, t_phase), dim=1)
    u = net_u(net_in).view(-1, 1)
    v = net_v(net_in).view(-1, 1)
    w = net_w(net_in).view(-1, 1)

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

    inv_cs = 1.0 / coord_scale

    # ── Viscosity ratio at wall ────────────────────────────────────────────
    if non_newtonian is not None:
        gamma_std = compute_shear_rate_std(u_x, u_y, u_z,
                                           v_x, v_y, v_z,
                                           w_x, w_y, w_z)
        shear_scale = U_ref / (L_ref * coord_scale)
        gamma_phys = gamma_std.detach() * shear_scale

        mu_ratio = carreau_yasuda_mu_ratio(
            gamma_phys,
            mu_0=non_newtonian['mu_0'],
            mu_inf=non_newtonian['mu_inf'],
            lam=non_newtonian['lambda'],
            n=non_newtonian['n'],
            a=non_newtonian['a'],
        )   # (N, 1), >= 1.0
    else:
        mu_ratio = 1.0  # scalar — Newtonian (cancels)

    # ── Non-dim stress tensor with viscosity correction ────────────────────
    # tau_bar_ij = mu_ratio * (1/cs) * (du_bar_i/dx_std_j + du_bar_j/dx_std_i)
    tau_xx = mu_ratio * 2.0 * u_x * inv_cs
    tau_yy = mu_ratio * 2.0 * v_y * inv_cs
    tau_zz = mu_ratio * 2.0 * w_z * inv_cs

    tau_xy = mu_ratio * (u_y + v_x) * inv_cs
    tau_xz = mu_ratio * (u_z + w_x) * inv_cs
    tau_yz = mu_ratio * (v_z + w_y) * inv_cs

    n_x = normals[:, 0:1]
    n_y = normals[:, 1:2]
    n_z = normals[:, 2:3]

    # Traction vector: t = tau_bar . n
    t_x = tau_xx * n_x + tau_xy * n_y + tau_xz * n_z
    t_y = tau_xy * n_x + tau_yy * n_y + tau_yz * n_z
    t_z = tau_xz * n_x + tau_yz * n_y + tau_zz * n_z

    t_dot_n = t_x * n_x + t_y * n_y + t_z * n_z

    # WSS = tangential component (in tau_bar units)
    wss_x_tau = t_x - t_dot_n * n_x
    wss_y_tau = t_y - t_dot_n * n_y
    wss_z_tau = t_z - t_dot_n * n_z

    # Convert tau_bar -> standardised units to match targets
    wss_x_pred = wss_x_tau / wss_std
    wss_y_pred = wss_y_tau / wss_std
    wss_z_pred = wss_z_tau / wss_std

    loss_fn = nn.MSELoss()
    loss = (loss_fn(wss_x_pred, wss_x_true)
            + loss_fn(wss_y_pred, wss_y_true)
            + loss_fn(wss_z_pred, wss_z_true))

    wss_pred = torch.cat([wss_x_pred, wss_y_pred, wss_z_pred], dim=1)

    return loss, wss_pred


def compute_wss_magnitude(wss_x, wss_y, wss_z):
    """Compute WSS magnitude from components."""
    return torch.sqrt(wss_x**2 + wss_y**2 + wss_z**2 + 1e-12)


def compute_wss_metrics(wss_pred, wss_true):
    """Compute validation metrics for WSS prediction."""
    numerator = torch.norm(wss_pred - wss_true)
    denominator = torch.norm(wss_true)
    relative_l2 = (numerator / (denominator + 1e-12)).item()

    pointwise_error = torch.abs(wss_pred - wss_true) / (torch.abs(wss_true) + 1e-12)
    mean_pointwise = pointwise_error.mean().item()
    max_pointwise = pointwise_error.max().item()

    wss_pred_flat = wss_pred.flatten()
    wss_true_flat = wss_true.flatten()

    vx = wss_pred_flat - wss_pred_flat.mean()
    vy = wss_true_flat - wss_true_flat.mean()
    correlation = (vx * vy).sum() / (torch.sqrt((vx**2).sum()) * torch.sqrt((vy**2).sum()) + 1e-12)
    correlation = correlation.item()

    mae = torch.abs(wss_pred - wss_true).mean().item()
    rmse = torch.sqrt(((wss_pred - wss_true)**2).mean()).item()

    return {
        'relative_l2': relative_l2,
        'mean_pointwise': mean_pointwise,
        'max_pointwise': max_pointwise,
        'correlation': correlation,
        'mae': mae,
        'rmse': rmse,
    }
