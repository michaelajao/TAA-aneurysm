"""
Physics Loss Functions — RANS Momentum + Continuity

Enforces the Reynolds-Averaged Navier-Stokes (RANS) equations for
incompressible flow with a *learnable* turbulent viscosity field nu_t(x).

The CFD training data was generated using the SST k-omega Transition model,
so the physics constraint must match: we enforce RANS (not laminar N-S).

Coordinate system note
----------------------
Training inputs are in STANDARDISED coordinates:
    x_std = (x - x_mean) / (L_ref * coord_scale)    in [-1, 1]

The network outputs:
    u, v, w  — non-dimensional velocity   u_bar = u / U_ref
    p        — standardised pressure      p_std = p_bar / pressure_std
    nut      — non-dimensional turbulent viscosity  nut_bar = nu_t / (U_ref * L_ref)

Autograd gives derivatives w.r.t. x_std.  The chain rule gives:
    d(u_bar)/d(x_bar) = (1/cs) * d(u_bar)/d(x_std)
    d^2(u_bar)/d(x_bar)^2 = (1/cs^2) * d^2(u_bar)/d(x_std)^2

RANS momentum in non-dimensional x_bar space:
    u . grad(u) + grad(p) = div[ nu_eff * (grad(u) + grad(u)^T) ]
    where nu_eff = 1/Re + nut_bar

For incompressible flow (div u = 0), the viscous term simplifies to:
    div[ nu_eff * (grad u + grad u^T) ]_i
        = nu_eff * laplacian(u_i)
          + d(nut_bar)/d(x_j) * [ d(u_i)/d(x_j) + d(u_j)/d(x_i) ]

Converting to x_std coords and multiplying through by coord_scale:

    u * du/dx_std + ... + pressure_std * dp/dx_std
        = (1/cs) * [ nu_eff * laplacian_std(u)
                     + d(nut)/dx_std_j * (du_i/dx_std_j + du_j/dx_std_i) ]

Continuity is unchanged in form (coord_scale cancels):
    du/dx_std + dv/dy_std + dw/dz_std = 0
"""

import torch
import torch.nn as nn


def compute_physics_loss(net_u, net_v, net_w, net_p, net_nut,
                        x, y, z, t_phase,
                        Re,
                        coord_scale: float = 1.0,
                        pressure_std: float = 1.0):
    """
    Compute RANS physics loss with learnable turbulent viscosity.

    Args:
        net_u, net_v, net_w, net_p: velocity/pressure networks
        net_nut: turbulent viscosity network (outputs non-dim nu_t >= 0)
        x, y, z:        Standardised collocation coordinates (N, 1)
        t_phase:        Cardiac phase (N, 1)
        Re:             Reynolds number  rho * U_ref * L_ref / mu
        coord_scale:    x_bar = x_std * coord_scale
        pressure_std:   p_bar = p_std * pressure_std

    Returns:
        loss:      Total physics loss (momentum + continuity)
        residuals: Dict of mean-absolute residuals for monitoring
    """
    inv_cs = 1.0 / coord_scale
    inv_Re = 1.0 / Re

    x = x.clone().detach().requires_grad_(True)
    y = y.clone().detach().requires_grad_(True)
    z = z.clone().detach().requires_grad_(True)

    net_in = torch.cat((x, y, z, t_phase), dim=1)
    u = net_u(net_in).view(-1, 1)
    v = net_v(net_in).view(-1, 1)
    w = net_w(net_in).view(-1, 1)
    p = net_p(net_in).view(-1, 1)
    nut = net_nut(net_in).view(-1, 1)  # non-dim turbulent viscosity (>= 0 via softplus)

    ones = torch.ones_like(u)

    # ── First derivatives of velocity w.r.t. x_std ────────────────────────
    u_x = torch.autograd.grad(u, x, ones, create_graph=True, retain_graph=True)[0]
    u_y = torch.autograd.grad(u, y, ones, create_graph=True, retain_graph=True)[0]
    u_z = torch.autograd.grad(u, z, ones, create_graph=True, retain_graph=True)[0]

    v_x = torch.autograd.grad(v, x, ones, create_graph=True, retain_graph=True)[0]
    v_y = torch.autograd.grad(v, y, ones, create_graph=True, retain_graph=True)[0]
    v_z = torch.autograd.grad(v, z, ones, create_graph=True, retain_graph=True)[0]

    w_x = torch.autograd.grad(w, x, ones, create_graph=True, retain_graph=True)[0]
    w_y = torch.autograd.grad(w, y, ones, create_graph=True, retain_graph=True)[0]
    w_z = torch.autograd.grad(w, z, ones, create_graph=True, retain_graph=True)[0]

    # ── Pressure gradients w.r.t. x_std ──────────────────────────────────
    p_x = torch.autograd.grad(p, x, ones, create_graph=True, retain_graph=True)[0]
    p_y = torch.autograd.grad(p, y, ones, create_graph=True, retain_graph=True)[0]
    p_z = torch.autograd.grad(p, z, ones, create_graph=True, retain_graph=True)[0]

    # ── Gradients of nu_t w.r.t. x_std ───────────────────────────────────
    nut_x = torch.autograd.grad(nut, x, ones, create_graph=True, retain_graph=True)[0]
    nut_y = torch.autograd.grad(nut, y, ones, create_graph=True, retain_graph=True)[0]
    nut_z = torch.autograd.grad(nut, z, ones, create_graph=True, retain_graph=True)[0]

    # ── Second derivatives (Laplacian components) ─────────────────────────
    u_xx = torch.autograd.grad(u_x, x, ones, create_graph=True, retain_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, ones, create_graph=True, retain_graph=True)[0]
    u_zz = torch.autograd.grad(u_z, z, ones, create_graph=True, retain_graph=True)[0]

    v_xx = torch.autograd.grad(v_x, x, ones, create_graph=True, retain_graph=True)[0]
    v_yy = torch.autograd.grad(v_y, y, ones, create_graph=True, retain_graph=True)[0]
    v_zz = torch.autograd.grad(v_z, z, ones, create_graph=True, retain_graph=True)[0]

    w_xx = torch.autograd.grad(w_x, x, ones, create_graph=True, retain_graph=True)[0]
    w_yy = torch.autograd.grad(w_y, y, ones, create_graph=True, retain_graph=True)[0]
    w_zz = torch.autograd.grad(w_z, z, ones, create_graph=True, retain_graph=True)[0]

    # ── Effective viscosity (non-dimensional) ─────────────────────────────
    nu_eff = inv_Re + nut  # (N, 1)

    # ── RANS Momentum residuals in standardised coords ────────────────────
    # Viscous: (1/cs) * [ nu_eff * laplacian_std(u_i)
    #            + d(nut)/dx_std_j * (du_i/dx_std_j + du_j/dx_std_i) ]

    # x-momentum
    laplacian_u = u_xx + u_yy + u_zz
    strain_dot_grad_nut_x = (nut_x * 2.0 * u_x
                             + nut_y * (u_y + v_x)
                             + nut_z * (u_z + w_x))
    viscous_x = inv_cs * (nu_eff * laplacian_u + strain_dot_grad_nut_x)

    residual_x = (u * u_x + v * u_y + w * u_z
                  + pressure_std * p_x
                  - viscous_x)

    # y-momentum
    laplacian_v = v_xx + v_yy + v_zz
    strain_dot_grad_nut_y = (nut_x * (v_x + u_y)
                             + nut_y * 2.0 * v_y
                             + nut_z * (v_z + w_y))
    viscous_y = inv_cs * (nu_eff * laplacian_v + strain_dot_grad_nut_y)

    residual_y = (u * v_x + v * v_y + w * v_z
                  + pressure_std * p_y
                  - viscous_y)

    # z-momentum
    laplacian_w = w_xx + w_yy + w_zz
    strain_dot_grad_nut_z = (nut_x * (w_x + u_z)
                             + nut_y * (w_y + v_z)
                             + nut_z * 2.0 * w_z)
    viscous_z = inv_cs * (nu_eff * laplacian_w + strain_dot_grad_nut_z)

    residual_z = (u * w_x + v * w_y + w * w_z
                  + pressure_std * p_z
                  - viscous_z)

    # ── Continuity ────────────────────────────────────────────────────────
    residual_continuity = u_x + v_y + w_z

    # ── MSE losses ────────────────────────────────────────────────────────
    loss_fn = nn.MSELoss()
    loss_x = loss_fn(residual_x, torch.zeros_like(residual_x))
    loss_y = loss_fn(residual_y, torch.zeros_like(residual_y))
    loss_z = loss_fn(residual_z, torch.zeros_like(residual_z))
    loss_cont = loss_fn(residual_continuity, torch.zeros_like(residual_continuity))

    loss = loss_x + loss_y + loss_z + loss_cont

    residuals = {
        'momentum_x': residual_x.detach().abs().mean().item(),
        'momentum_y': residual_y.detach().abs().mean().item(),
        'momentum_z': residual_z.detach().abs().mean().item(),
        'continuity': residual_continuity.detach().abs().mean().item(),
        'nut_mean': nut.detach().mean().item(),
        'nut_max': nut.detach().max().item(),
    }

    return loss, residuals


def compute_continuity_loss(net_u, net_v, net_w,
                            x, y, z, t_phase):
    """
    Compute only the non-dimensional continuity equation loss.

    Continuity: div(u_bar) = 0

    Args:
        net_u, net_v, net_w: Neural networks for non-dim velocity
        x, y, z: Non-dimensional coordinates (N, 1) with requires_grad=True
        t_phase: Cardiac phase (N, 1)

    Returns:
        loss: Continuity loss
    """
    x = x.clone().detach().requires_grad_(True)
    y = y.clone().detach().requires_grad_(True)
    z = z.clone().detach().requires_grad_(True)

    net_in = torch.cat((x, y, z, t_phase), dim=1)
    u = net_u(net_in).view(-1, 1)
    v = net_v(net_in).view(-1, 1)
    w = net_w(net_in).view(-1, 1)

    ones = torch.ones_like(u)
    u_x = torch.autograd.grad(u, x, ones, create_graph=True, retain_graph=True)[0]
    v_y = torch.autograd.grad(v, y, ones, create_graph=True, retain_graph=True)[0]
    w_z = torch.autograd.grad(w, z, ones, create_graph=True, retain_graph=True)[0]

    divergence = u_x + v_y + w_z

    loss_fn = nn.MSELoss()
    loss = loss_fn(divergence, torch.zeros_like(divergence))

    return loss
