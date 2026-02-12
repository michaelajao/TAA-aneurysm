"""
Physics Loss Functions - Navier-Stokes Equations

Enforces conservation of momentum and mass for incompressible Newtonian fluid.
"""

import torch
import torch.nn as nn


def compute_physics_loss(net_u, net_v, net_w, net_p,
                        x, y, z, t_phase,
                        rho=1.0,
                        mu=0.00125,
                        X_scale=1.0,
                        Y_scale=1.0,
                        Z_scale=1.0,
                        U_scale=1.0):
    """
    Compute physics loss from Navier-Stokes equations.

    Steady-state incompressible Navier-Stokes:
        X-momentum: u·∂u/∂x + v·∂u/∂y + w·∂u/∂z = ν(∂²u/∂x² + ∂²u/∂y² + ∂²u/∂z²) - (1/ρ)·∂p/∂x
        Y-momentum: u·∂v/∂x + v·∂v/∂y + w·∂v/∂z = ν(∂²v/∂x² + ∂²v/∂y² + ∂²v/∂z²) - (1/ρ)·∂p/∂y
        Z-momentum: u·∂w/∂x + v·∂w/∂y + w·∂w/∂z = ν(∂²w/∂x² + ∂²w/∂y² + ∂²w/∂z²) - (1/ρ)·∂p/∂z
        Continuity: ∂u/∂x + ∂v/∂y + ∂w/∂z = 0

    where ν = μ/ρ is kinematic viscosity.

    Args:
        net_u, net_v, net_w, net_p: Neural networks
        x, y, z: Collocation point coordinates (N, 1) with requires_grad=True
        t_phase: Cardiac phase (N, 1)
        rho: Fluid density
        mu: Dynamic viscosity
        X_scale, Y_scale, Z_scale: Coordinate scaling
        U_scale: Velocity scaling

    Returns:
        loss: Total physics loss (momentum + continuity)
        residuals: Dictionary of residuals for monitoring
    """
    # Kinematic viscosity
    nu = mu / rho

    # Ensure gradients are enabled
    x = x.clone().detach().requires_grad_(True)
    y = y.clone().detach().requires_grad_(True)
    z = z.clone().detach().requires_grad_(True)

    # Forward pass
    net_in = torch.cat((x, y, z, t_phase), dim=1)
    u = net_u(net_in).view(-1, 1)
    v = net_v(net_in).view(-1, 1)
    w = net_w(net_in).view(-1, 1)
    p = net_p(net_in).view(-1, 1)

    # First derivatives
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                              create_graph=True, retain_graph=True)[0]
    u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u),
                              create_graph=True, retain_graph=True)[0]
    u_z = torch.autograd.grad(u, z, grad_outputs=torch.ones_like(u),
                              create_graph=True, retain_graph=True)[0]

    v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v),
                              create_graph=True, retain_graph=True)[0]
    v_y = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(v),
                              create_graph=True, retain_graph=True)[0]
    v_z = torch.autograd.grad(v, z, grad_outputs=torch.ones_like(v),
                              create_graph=True, retain_graph=True)[0]

    w_x = torch.autograd.grad(w, x, grad_outputs=torch.ones_like(w),
                              create_graph=True, retain_graph=True)[0]
    w_y = torch.autograd.grad(w, y, grad_outputs=torch.ones_like(w),
                              create_graph=True, retain_graph=True)[0]
    w_z = torch.autograd.grad(w, z, grad_outputs=torch.ones_like(w),
                              create_graph=True, retain_graph=True)[0]

    p_x = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(p),
                              create_graph=True, retain_graph=True)[0]
    p_y = torch.autograd.grad(p, y, grad_outputs=torch.ones_like(p),
                              create_graph=True, retain_graph=True)[0]
    p_z = torch.autograd.grad(p, z, grad_outputs=torch.ones_like(p),
                              create_graph=True, retain_graph=True)[0]

    # Second derivatives
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x),
                               create_graph=True, retain_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y),
                               create_graph=True, retain_graph=True)[0]
    u_zz = torch.autograd.grad(u_z, z, grad_outputs=torch.ones_like(u_z),
                               create_graph=True, retain_graph=True)[0]

    v_xx = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(v_x),
                               create_graph=True, retain_graph=True)[0]
    v_yy = torch.autograd.grad(v_y, y, grad_outputs=torch.ones_like(v_y),
                               create_graph=True, retain_graph=True)[0]
    v_zz = torch.autograd.grad(v_z, z, grad_outputs=torch.ones_like(v_z),
                               create_graph=True, retain_graph=True)[0]

    w_xx = torch.autograd.grad(w_x, x, grad_outputs=torch.ones_like(w_x),
                               create_graph=True, retain_graph=True)[0]
    w_yy = torch.autograd.grad(w_y, y, grad_outputs=torch.ones_like(w_y),
                               create_graph=True, retain_graph=True)[0]
    w_zz = torch.autograd.grad(w_z, z, grad_outputs=torch.ones_like(w_z),
                               create_graph=True, retain_graph=True)[0]

    # Scaling factors (from original 3D PINN-wss code)
    XX_scale = U_scale * (X_scale ** 2)
    YY_scale = U_scale * (Y_scale ** 2)
    ZZ_scale = U_scale * (Z_scale ** 2)
    UU_scale = U_scale ** 2

    # X-momentum residual
    residual_x = (u * u_x / X_scale +
                  v * u_y / Y_scale +
                  w * u_z / Z_scale -
                  nu * (u_xx / XX_scale + u_yy / YY_scale + u_zz / ZZ_scale) +
                  (1 / rho) * (p_x / (X_scale * UU_scale)))

    # Y-momentum residual
    residual_y = (u * v_x / X_scale +
                  v * v_y / Y_scale +
                  w * v_z / Z_scale -
                  nu * (v_xx / XX_scale + v_yy / YY_scale + v_zz / ZZ_scale) +
                  (1 / rho) * (p_y / (Y_scale * UU_scale)))

    # Z-momentum residual
    residual_z = (u * w_x / X_scale +
                  v * w_y / Y_scale +
                  w * w_z / Z_scale -
                  nu * (w_xx / XX_scale + w_yy / YY_scale + w_zz / ZZ_scale) +
                  (1 / rho) * (p_z / (Z_scale * UU_scale)))

    # Continuity residual
    residual_continuity = u_x / X_scale + v_y / Y_scale + w_z / Z_scale

    # MSE loss for each residual (target is zero)
    loss_fn = nn.MSELoss()
    loss_x = loss_fn(residual_x, torch.zeros_like(residual_x))
    loss_y = loss_fn(residual_y, torch.zeros_like(residual_y))
    loss_z = loss_fn(residual_z, torch.zeros_like(residual_z))
    loss_cont = loss_fn(residual_continuity, torch.zeros_like(residual_continuity))

    # Total physics loss
    loss = loss_x + loss_y + loss_z + loss_cont

    # Residuals for monitoring
    residuals = {
        'momentum_x': residual_x.detach().abs().mean().item(),
        'momentum_y': residual_y.detach().abs().mean().item(),
        'momentum_z': residual_z.detach().abs().mean().item(),
        'continuity': residual_continuity.detach().abs().mean().item()
    }

    return loss, residuals


def compute_continuity_loss(net_u, net_v, net_w,
                            x, y, z, t_phase,
                            X_scale=1.0,
                            Y_scale=1.0,
                            Z_scale=1.0):
    """
    Compute only the continuity equation loss.

    Continuity: ∂u/∂x + ∂v/∂y + ∂w/∂z = 0

    This can be used separately from momentum equations if needed.

    Args:
        net_u, net_v, net_w: Neural networks for velocity
        x, y, z: Coordinates (N, 1) with requires_grad=True
        t_phase: Cardiac phase (N, 1)
        X_scale, Y_scale, Z_scale: Coordinate scaling

    Returns:
        loss: Continuity loss
    """
    # Ensure gradients are enabled
    x = x.clone().detach().requires_grad_(True)
    y = y.clone().detach().requires_grad_(True)
    z = z.clone().detach().requires_grad_(True)

    # Forward pass
    net_in = torch.cat((x, y, z, t_phase), dim=1)
    u = net_u(net_in).view(-1, 1)
    v = net_v(net_in).view(-1, 1)
    w = net_w(net_in).view(-1, 1)

    # Compute divergence
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                              create_graph=True, retain_graph=True)[0]
    v_y = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(v),
                              create_graph=True, retain_graph=True)[0]
    w_z = torch.autograd.grad(w, z, grad_outputs=torch.ones_like(w),
                              create_graph=True, retain_graph=True)[0]

    # Divergence with scaling
    divergence = u_x / X_scale + v_y / Y_scale + w_z / Z_scale

    # MSE loss (target is zero)
    loss_fn = nn.MSELoss()
    loss = loss_fn(divergence, torch.zeros_like(divergence))

    return loss
