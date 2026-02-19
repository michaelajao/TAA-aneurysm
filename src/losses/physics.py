"""
Physics Loss Functions — RANS Momentum + Continuity

Enforces the Reynolds-Averaged Navier-Stokes (RANS) equations for
incompressible flow with a *learnable* turbulent viscosity field nu_t(x)
and optional Carreau-Yasuda non-Newtonian molecular viscosity.

The CFD training data was generated using the SST k-omega Transition model
with Carreau-Yasuda blood rheology, so the physics constraint must match.

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
    where nu_eff = nu_mol(gamma_dot) + nut_bar
          nu_mol = mu(gamma_dot) / (rho * U_ref * L_ref) = mu_ratio / Re
          mu_ratio = mu(gamma_dot) / mu_inf   (from Carreau-Yasuda)

For incompressible flow (div u = 0), the viscous term simplifies to:
    div[ nu_eff * (grad u + grad u^T) ]_i
        = nu_eff * laplacian(u_i)
          + d(nu_eff)/d(x_j) * [ d(u_i)/d(x_j) + d(u_j)/d(x_i) ]

When mu varies spatially (non-Newtonian), grad(nu_eff) includes contributions
from both grad(nu_mol) and grad(nut).  Since nu_mol depends on the velocity
gradients via the shear rate, computing grad(nu_mol) analytically through
autograd would require third derivatives and is prohibitively expensive.

Instead we use a quasi-steady approximation: treat nu_mol as a spatially
varying coefficient (detached from the autograd graph for its own gradients)
and only track grad(nut) explicitly.  This is standard practice in RANS
solvers where the turbulence model viscosity is lagged by one iteration.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple


def carreau_yasuda_mu_ratio(gamma_dot_phys: torch.Tensor,
                            mu_0: float = 0.16,
                            mu_inf: float = 0.0035,
                            lam: float = 8.2,
                            n: float = 0.2128,
                            a: float = 0.64) -> torch.Tensor:
    """Compute viscosity ratio mu(gamma_dot)/mu_inf using Carreau-Yasuda model.

    mu(gamma_dot) = mu_inf + (mu_0 - mu_inf) * (1 + (lam * gamma_dot)^a)^((n-1)/a)

    Returns mu/mu_inf so it can multiply 1/Re directly.  When gamma_dot is
    large the ratio approaches 1.0 (Newtonian limit).

    Args:
        gamma_dot_phys: Shear rate magnitude in physical units (s^-1), shape (N,1)
        mu_0:   Zero-shear viscosity (Pa.s)
        mu_inf: Infinite-shear viscosity (Pa.s)
        lam:    Relaxation time (s)
        n:      Power-law index
        a:      Yasuda parameter

    Returns:
        mu_ratio: mu(gamma_dot)/mu_inf, shape (N,1), >= 1.0
    """
    exponent = (n - 1.0) / a
    mu = mu_inf + (mu_0 - mu_inf) * (1.0 + (lam * gamma_dot_phys).pow(a)).pow(exponent)
    return mu / mu_inf


def compute_shear_rate_std(u_x, u_y, u_z,
                           v_x, v_y, v_z,
                           w_x, w_y, w_z) -> torch.Tensor:
    """Compute shear rate magnitude from velocity gradients in x_std coords.

    gamma_dot_std = sqrt(2 * S_ij_std * S_ij_std)
    where S_ij_std = 0.5 * (du_i/dx_std_j + du_j/dx_std_i)

    The physical shear rate is:
        gamma_dot_phys = (U_ref / (L_ref * coord_scale)) * gamma_dot_std
    """
    s_xx = u_x
    s_yy = v_y
    s_zz = w_z
    s_xy = 0.5 * (u_y + v_x)
    s_xz = 0.5 * (u_z + w_x)
    s_yz = 0.5 * (v_z + w_y)

    # 2 * S_ij * S_ij = 2*(s_xx^2 + s_yy^2 + s_zz^2 + 2*s_xy^2 + 2*s_xz^2 + 2*s_yz^2)
    two_SijSij = 2.0 * (s_xx**2 + s_yy**2 + s_zz**2
                        + 2.0 * (s_xy**2 + s_xz**2 + s_yz**2))
    return torch.sqrt(two_SijSij + 1e-12)


def compute_physics_loss(net_u, net_v, net_w, net_p, net_nut,
                        x, y, z, t_phase,
                        Re,
                        coord_scale: float = 1.0,
                        pressure_std: float = 1.0,
                        non_newtonian: Optional[Dict] = None,
                        U_ref: float = 1.0,
                        L_ref: float = 1.0):
    """
    Compute RANS physics loss with learnable turbulent viscosity and
    optional Carreau-Yasuda non-Newtonian molecular viscosity.

    Args:
        net_u, net_v, net_w, net_p: velocity/pressure networks
        net_nut: turbulent viscosity network (outputs non-dim nu_t >= 0)
        x, y, z:        Standardised collocation coordinates (N, 1)
        t_phase:        Cardiac phase (N, 1)
        Re:             Reynolds number  rho * U_ref * L_ref / mu_inf
        coord_scale:    x_bar = x_std * coord_scale
        pressure_std:   p_bar = p_std * pressure_std
        non_newtonian:  Dict with Carreau-Yasuda params, or None for Newtonian
                        Keys: mu_0, mu_inf, lambda, n, a (all in physical units)
        U_ref:          Reference velocity (m/s), needed for shear rate scaling
        L_ref:          Reference length (m), needed for shear rate scaling

    Returns:
        loss:      Total physics loss (momentum + continuity)
        residuals: Dict of mean-absolute residuals for monitoring
    """
    inv_cs = 1.0 / coord_scale

    x = x.clone().detach().requires_grad_(True)
    y = y.clone().detach().requires_grad_(True)
    z = z.clone().detach().requires_grad_(True)

    net_in = torch.cat((x, y, z, t_phase), dim=1)
    u = net_u(net_in).view(-1, 1)
    v = net_v(net_in).view(-1, 1)
    w = net_w(net_in).view(-1, 1)
    p = net_p(net_in).view(-1, 1)
    nut = net_nut(net_in).view(-1, 1)

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

    # ── Molecular viscosity (non-dimensional) ──────────────────────────────
    inv_Re = 1.0 / Re

    if non_newtonian is not None:
        # Shear rate in x_std coords → physical units
        gamma_std = compute_shear_rate_std(u_x, u_y, u_z,
                                           v_x, v_y, v_z,
                                           w_x, w_y, w_z)
        # gamma_phys = (U_ref / (L_ref * coord_scale)) * gamma_std
        shear_scale = U_ref / (L_ref * coord_scale)
        gamma_phys = gamma_std.detach() * shear_scale  # detach: quasi-steady approx

        mu_ratio = carreau_yasuda_mu_ratio(
            gamma_phys,
            mu_0=non_newtonian['mu_0'],
            mu_inf=non_newtonian['mu_inf'],
            lam=non_newtonian['lambda'],
            n=non_newtonian['n'],
            a=non_newtonian['a'],
        )
        nu_mol = inv_Re * mu_ratio  # (N, 1), spatially varying
    else:
        nu_mol = inv_Re  # scalar, constant (Newtonian)
        mu_ratio = None

    # ── Effective viscosity ─────────────────────────────────────────────────
    nu_eff = nu_mol + nut  # (N, 1)

    # ── RANS Momentum residuals in standardised coords ────────────────────
    # With non-Newtonian viscosity, nu_mol varies in space but we treat its
    # spatial gradient as zero (quasi-steady).  Only grad(nut) contributes
    # to the grad(nu_eff) · strain term.

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

    if mu_ratio is not None:
        residuals['mu_ratio_mean'] = mu_ratio.mean().item()
        residuals['mu_ratio_max'] = mu_ratio.max().item()

    return loss, residuals
