"""
Physics Loss Functions - Non-Dimensional Navier-Stokes Equations

Enforces conservation of momentum and mass for incompressible Newtonian fluid
in fully non-dimensional form:

    Momentum:   u_bar . grad(u_bar) = -grad(p_bar) + (1/Re) * laplacian(u_bar)
    Continuity: div(u_bar) = 0

where Re = rho * U_ref * L_ref / mu is the Reynolds number.

All inputs (coordinates, velocities, pressure) are assumed to be non-dimensional.
"""

import torch
import torch.nn as nn


def compute_physics_loss(net_u, net_v, net_w, net_p,
                        x, y, z, t_phase,
                        Re):
    """
    Compute physics loss from the non-dimensional Navier-Stokes equations.

    Args:
        net_u, net_v, net_w, net_p: Neural networks for non-dim velocity and pressure
        x, y, z: Non-dimensional collocation coordinates (N, 1) with requires_grad=True
        t_phase: Cardiac phase (N, 1)
        Re: Reynolds number (float)

    Returns:
        loss: Total physics loss (momentum + continuity)
        residuals: Dictionary of residuals for monitoring
    """
    inv_Re = 1.0 / Re

    # Ensure gradients are enabled
    x = x.clone().detach().requires_grad_(True)
    y = y.clone().detach().requires_grad_(True)
    z = z.clone().detach().requires_grad_(True)

    # Forward pass (outputs are non-dimensional)
    net_in = torch.cat((x, y, z, t_phase), dim=1)
    u = net_u(net_in).view(-1, 1)
    v = net_v(net_in).view(-1, 1)
    w = net_w(net_in).view(-1, 1)
    p = net_p(net_in).view(-1, 1)

    # First derivatives (all non-dimensional: d(u_bar)/d(x_bar))
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

    # Non-dimensional momentum residuals:
    #   u . grad(u) + grad(p) - (1/Re) * laplacian(u) = 0
    residual_x = (u * u_x + v * u_y + w * u_z
                  + p_x
                  - inv_Re * (u_xx + u_yy + u_zz))

    residual_y = (u * v_x + v * v_y + w * v_z
                  + p_y
                  - inv_Re * (v_xx + v_yy + v_zz))

    residual_z = (u * w_x + v * w_y + w * w_z
                  + p_z
                  - inv_Re * (w_xx + w_yy + w_zz))

    # Non-dimensional continuity residual: div(u) = 0
    residual_continuity = u_x + v_y + w_z

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
    # Ensure gradients are enabled
    x = x.clone().detach().requires_grad_(True)
    y = y.clone().detach().requires_grad_(True)
    z = z.clone().detach().requires_grad_(True)

    # Forward pass
    net_in = torch.cat((x, y, z, t_phase), dim=1)
    u = net_u(net_in).view(-1, 1)
    v = net_v(net_in).view(-1, 1)
    w = net_w(net_in).view(-1, 1)

    # Compute divergence (non-dimensional, no scaling needed)
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                              create_graph=True, retain_graph=True)[0]
    v_y = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(v),
                              create_graph=True, retain_graph=True)[0]
    w_z = torch.autograd.grad(w, z, grad_outputs=torch.ones_like(w),
                              create_graph=True, retain_graph=True)[0]

    divergence = u_x + v_y + w_z

    # MSE loss (target is zero)
    loss_fn = nn.MSELoss()
    loss = loss_fn(divergence, torch.zeros_like(divergence))

    return loss
