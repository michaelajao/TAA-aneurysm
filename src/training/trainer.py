"""
Main Training Script for Single Geometry TAA-PINN

Integrates all components:
- Data loading
- Network initialization
- Loss functions
- Training loop
- Logging and checkpointing
"""

import csv as csv_mod
import os
import sys
import yaml
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

from src.data.loader import TAADataLoader
from src.utils.geometry import compute_wall_normals_torch, sample_interior_points_torch
from src.models.networks import create_taa_networks, count_parameters
from src.losses.wss import compute_wss_loss, compute_wss_metrics
from src.losses.physics import compute_physics_loss
from src.utils.plotting import process_geometry as generate_comparison_plots
from src.losses.boundary import (compute_noslip_loss, compute_pressure_loss,
                                 detect_inlet_outlet, generate_cross_section_points,
                                 compute_inlet_velocity_loss, compute_outlet_pressure_loss)

from conflictfree.grad_operator import ConFIGOperator  # type: ignore


class TAATrainer:
    """Trainer class for TAA-PINN."""

    def __init__(self, config_path, resume_checkpoint=None):
        """
        Initialize trainer from configuration file.

        Args:
            config_path: Path to YAML configuration file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        print("=" * 70)
        print("TAA-PINN TRAINING")
        print("=" * 70)
        print(f"\nExperiment: {self.config['experiment']['name']}")
        print(f"Description: {self.config['experiment']['description']}")
        print(f"Geometry: {self.config['data']['geometry']}")

        # Set random seed
        torch.manual_seed(self.config['random_seed'])
        np.random.seed(self.config['random_seed'])

        # Set device
        self.device = self.config['model']['device']
        if self.device == 'cuda' and not torch.cuda.is_available():
            print("CUDA not available, using CPU")
            self.device = 'cpu'

        print(f"Device: {self.device}")

        # Create output directory
        self.output_dir = Path(self.config['training']['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {self.output_dir}")

        # Training state
        self.epoch = 0
        self.best_loss = float('inf')
        self.loss_history = []  # list of dicts, one per epoch
        self.start_epoch = 1

        # Adaptive loss normalization: tracks initial raw loss magnitudes
        # and scales weights so all loss components contribute equally at start
        self.loss_normalization = self.config['training'].get('loss_normalization', False)
        self.renorm_interval = self.config['training'].get('renorm_interval', 0)
        self.loss_norms = {}  # populated after first epoch

        # Physics loss annealing: ramp lambda_physics from 0 to full weight
        self.physics_ramp_epochs = self.config['loss_weights'].get('physics_ramp_epochs', 0)

        # Dynamic collocation resampling interval
        self.resample_collocation_interval = self.config['physics'].get(
            'resample_collocation_interval', 0)

        # Early stopping configuration (tracks training loss — no data split in PINNs)
        early_cfg = self.config['training'].get('early_stopping', {})
        self.early_stopping_enabled = early_cfg.get('enabled', False)
        self.early_stopping_patience = int(early_cfg.get('patience', 2000))
        self.early_stopping_min_delta = float(early_cfg.get('min_delta', 1e-6))
        self.no_improve_count = 0

        # Adaptive weights: Wang et al. 2021 gradient-norm balancing
        aw_cfg = self.config.get('adaptive_weights', {})
        self.adaptive_weights_enabled = aw_cfg.get('enabled', False)
        self.adaptive_weights_alpha = aw_cfg.get('alpha', 0.1)
        self.adaptive_weights_update_interval = aw_cfg.get('update_interval', 100)
        self.adaptive_weights_ref_loss = aw_cfg.get('ref_loss', 'wss')
        self.physics_weight_floor = aw_cfg.get('physics_weight_floor', 0.0)
        self.adaptive_weights = {}  # {name: float} -- populated on first update
        self._last_grad_norms = {}  # for logging: most recent per-loss grad norms

        # Non-Newtonian viscosity config (Carreau-Yasuda)
        nn_cfg = self.config.get('physics', {}).get('non_newtonian', {})
        if nn_cfg.get('enabled', False):
            self.non_newtonian = {
                'mu_0': nn_cfg['mu_0'],
                'mu_inf': nn_cfg['mu_inf'],
                'lambda': nn_cfg['lambda'],
                'n': nn_cfg['n'],
                'a': nn_cfg['a'],
            }
            print(f"\nNon-Newtonian viscosity: Carreau-Yasuda")
            print(f"  mu_inf={nn_cfg['mu_inf']}, mu_0={nn_cfg['mu_0']}, "
                  f"lambda={nn_cfg['lambda']}, n={nn_cfg['n']}, a={nn_cfg['a']}")
        else:
            self.non_newtonian = None

        # Optimizer strategy: "adaptive_weights" (default) or "config" (ConFIG)
        self.optimizer_strategy = self.config.get('optimizer_strategy', 'adaptive_weights')
        if self.optimizer_strategy == 'config':
            self.config_operator = ConFIGOperator()
            print(f"Optimizer strategy: ConFIG (conflict-free gradient)")
        else:
            self.config_operator = None
            print(f"Optimizer strategy: adaptive weights")

        # Initialize components (after all config attributes are set)
        self._load_data()
        self._initialize_networks()
        self._initialize_optimizers()

        # Optional resume
        if resume_checkpoint:
            self.load_checkpoint(resume_checkpoint)

        print("\nInitialization complete!")

    def _load_data(self):
        """Load and prepare training data with non-dimensional reference scales."""
        print("\n" + "-" * 70)
        print("LOADING DATA (non-dimensional)")
        print("-" * 70)

        norm = self.config['data']['normalization']
        physics = self.config['physics']

        # Initialize data loader with fluid properties
        loader = TAADataLoader(
            data_dir=self.config['data']['data_dir'],
            L_ref=norm['length_scale'],
            rho=physics['rho'],
            mu=physics['mu'],
            device=self.device
        )

        # First pass: compute reference scales from all training files
        training_files = [self.config['data']['files'][p]
                          for p in self.config['data']['phases']]
        self.ref_scales = loader.compute_reference_scales(training_files)
        self.Re = self.ref_scales['Re']
        self.wss_std = self.ref_scales.get('wss_std', 1.0)

        # Load both phases (second pass: normalize using computed scales)
        self.data = {}
        for phase in self.config['data']['phases']:
            filename = self.config['data']['files'][phase]
            print(f"\nLoading {phase}: {filename}")

            data = loader.load_single_case(
                filename,
                subsample_factor=self.config['data']['subsample_factor']
            )

            # Prepare tensors
            tensors = loader.prepare_tensors(data, include_phase=True)

            # Compute wall normals
            print(f"Computing wall normals...")
            normals = compute_wall_normals_torch(
                tensors['x'], tensors['y'], tensors['z'],
                radius=self.config['geometry']['normal_estimation']['radius'],
                max_nn=self.config['geometry']['normal_estimation']['max_nn'],
                orient_inward=self.config['geometry']['normal_estimation']['orient_inward'],
                device=self.device
            )
            tensors['normals'] = normals

            # Sample interior points for physics loss
            print(f"Sampling interior collocation points...")
            x_int, y_int, z_int = sample_interior_points_torch(
                tensors['x'], tensors['y'], tensors['z'],
                n_samples=self.config['physics']['n_interior_points'],
                offset_range=self.config['physics']['interior_offset_range'],
                normals=normals,
                seed=self.config['random_seed'],
                device=self.device
            )

            tensors['x_interior'] = x_int
            tensors['y_interior'] = y_int
            tensors['z_interior'] = z_int
            # One phase value per interior point (all same for a given cardiac phase)
            phase_val = tensors['phase'][0].item()
            tensors['phase_interior'] = torch.full(
                (x_int.shape[0], 1), phase_val,
                dtype=torch.float32, device=self.device
            )

            self.data[phase] = tensors

            # Print statistics
            print(f"  Wall points: {tensors['x'].shape[0]}")
            print(f"  Interior points: {x_int.shape[0]}")
            print(f"  Normals computed: {normals.shape}")
            print(f"  Phase encoding: {tensors['phase'][0].item()}")

        # ── Generate inlet/outlet cross-section sample points ────────────
        inlet_outlet_cfg = self.config.get('inlet_outlet', {})
        if inlet_outlet_cfg.get('enabled', True):
            first_phase = self.config['data']['phases'][0]
            tensors0 = self.data[first_phase]
            io_geom = detect_inlet_outlet(tensors0['x'], tensors0['y'], tensors0['z'])
            self.io_geometry = io_geom

            # Store inlet velocities from config (physical m/s)
            inlet_vel = inlet_outlet_cfg.get('inlet_velocity', {})
            self.inlet_vel_systolic = inlet_vel.get('systolic', 0.5)
            self.inlet_vel_diastolic = inlet_vel.get('diastolic', 0.1)

            axial_dim = io_geom['axial_dim']
            L_ref = self.config['data']['normalization']['length_scale']
            coord_scale = self.ref_scales.get('coord_scale', 1.0)

            n_radial = inlet_outlet_cfg.get('n_radial', 6)
            n_angular = inlet_outlet_cfg.get('n_angular', 12)

            for label in ['inlet', 'outlet']:
                # Generate points in PHYSICAL coordinates
                x_io, y_io, z_io = generate_cross_section_points(
                    io_geom[f'{label}_axial_pos'],
                    io_geom[f'{label}_center'],
                    io_geom[f'{label}_radius'],
                    axial_dim,
                    n_radial=n_radial,
                    n_angular=n_angular,
                    device='cpu',
                )
                n_io = x_io.shape[0]
                phys_radius = io_geom[f'{label}_radius'] * coord_scale * L_ref
                print(f"  {label.capitalize()} cross-section: {n_io} points, "
                      f"radius={phys_radius*1000:.1f} mm")

                for phase in self.config['data']['phases']:
                    phase_val = 1.0 if phase == 'systolic' else 0.0
                    t_io = torch.full((n_io, 1), phase_val,
                                      dtype=torch.float32, device=self.device)
                    self.data[phase][f'x_{label}'] = x_io.to(self.device)
                    self.data[phase][f'y_{label}'] = y_io.to(self.device)
                    self.data[phase][f'z_{label}'] = z_io.to(self.device)
                    self.data[phase][f'phase_{label}'] = t_io

            print(f"  Axial direction: {'XYZ'[axial_dim]}")
            print(f"  Inlet velocity (physical): systolic={self.inlet_vel_systolic} m/s, "
                  f"diastolic={self.inlet_vel_diastolic} m/s")
        else:
            self.io_geometry = None
            print("  Inlet/outlet BCs: DISABLED")

        print(f"\nData loading complete!")

    def _resample_collocation_points(self):
        """Re-sample interior collocation points for physics loss.

        Supports partial replacement via ``physics.resample_fraction`` (0-1).
        A fraction of 1.0 replaces all points (original behaviour); values
        below 1.0 keep a random subset of existing points and only sample
        fresh replacements for the remainder, preventing abrupt loss spikes.
        """
        fraction = self.config['physics'].get('resample_fraction', 1.0)

        for phase in self.config['data']['phases']:
            tensors = self.data[phase]
            n_total = self.config['physics']['n_interior_points']

            if fraction >= 1.0:
                # Full replacement (original behaviour)
                x_int, y_int, z_int = sample_interior_points_torch(
                    tensors['x'], tensors['y'], tensors['z'],
                    n_samples=n_total,
                    offset_range=self.config['physics']['interior_offset_range'],
                    normals=tensors['normals'],
                    seed=None,
                    device=self.device
                )
            else:
                n_keep = int(n_total * (1.0 - fraction))
                n_new = n_total - n_keep

                # Keep a random subset of existing points
                perm = torch.randperm(tensors['x_interior'].shape[0],
                                      device=self.device)[:n_keep]
                x_keep = tensors['x_interior'][perm]
                y_keep = tensors['y_interior'][perm]
                z_keep = tensors['z_interior'][perm]

                # Sample fresh points
                x_new, y_new, z_new = sample_interior_points_torch(
                    tensors['x'], tensors['y'], tensors['z'],
                    n_samples=n_new,
                    offset_range=self.config['physics']['interior_offset_range'],
                    normals=tensors['normals'],
                    seed=None,
                    device=self.device
                )

                x_int = torch.cat([x_keep, x_new], dim=0)
                y_int = torch.cat([y_keep, y_new], dim=0)
                z_int = torch.cat([z_keep, z_new], dim=0)

            tensors['x_interior'] = x_int
            tensors['y_interior'] = y_int
            tensors['z_interior'] = z_int
            phase_val = tensors['phase'][0].item()
            tensors['phase_interior'] = torch.full(
                (x_int.shape[0], 1), phase_val,
                dtype=torch.float32, device=self.device
            )

    def _initialize_networks(self):
        """Initialize neural networks."""
        print("\n" + "-" * 70)
        print("INITIALIZING NETWORKS")
        print("-" * 70)

        nut_cfg = self.config['model'].get('nut', {})
        self.networks = create_taa_networks(
            input_dim=self.config['model']['input_dim'],
            hidden_dim=self.config['model']['hidden_dim'],
            num_layers=self.config['model']['num_layers'],
            num_frequencies=self.config['model']['num_frequencies'],
            fourier_scale=self.config['model']['fourier_scale'],
            use_fourier=self.config['model']['use_fourier'],
            nut_hidden_dim=nut_cfg.get('hidden_dim', 64),
            nut_num_layers=nut_cfg.get('num_layers', 4),
            nu_t_min=nut_cfg.get('nu_t_min', 0.001),
            device=self.device
        )

        # Print network information
        for name, net in self.networks.items():
            n_params = count_parameters(net)
            print(f"  Net_{name}: {n_params:,} parameters")

        total_params = sum(count_parameters(net) for net in self.networks.values())
        print(f"  Total: {total_params:,} parameters")

    def _initialize_optimizers(self):
        """Initialize optimizers: one for the flow networks (u,v,w,p) and one
        for the turbulent viscosity network (nut).

        Alternating optimisation: the flow optimizer updates u/v/w/p using all
        losses; the nut optimizer updates net_nut using only the physics loss.
        This ensures net_nut receives dedicated gradient signal to learn the
        spatially-varying turbulent viscosity.
        """
        print("\n" + "-" * 70)
        print("INITIALIZING OPTIMIZERS")
        print("-" * 70)

        lr = self.config['training']['learning_rate']
        nut_lr_mult = self.config['model'].get('nut', {}).get('lr_multiplier', 10.0)
        nut_lr = lr * nut_lr_mult
        print(f"Learning rate (flow): {lr}")
        print(f"Learning rate (nut):  {nut_lr} ({nut_lr_mult}x)")

        flow_params = []
        for name, net in self.networks.items():
            if name != 'nut':
                flow_params += list(net.parameters())

        self.optimizer = optim.AdamW(
            flow_params,
            lr=lr,
            betas=(0.9, 0.99),
            eps=1e-15,
            weight_decay=1e-4,
        )

        nut_params = list(self.networks['nut'].parameters())
        self.optimizer_nut = optim.AdamW(
            nut_params,
            lr=nut_lr,
            betas=(0.9, 0.99),
            eps=1e-15,
            weight_decay=1e-5,
        )

        sched_cfg = self.config['training']['scheduler']
        sched_type = sched_cfg['type']
        if sched_type == 'StepLR':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=sched_cfg['step_size'],
                gamma=sched_cfg['gamma']
            )
            self.scheduler_nut = optim.lr_scheduler.StepLR(
                self.optimizer_nut,
                step_size=sched_cfg['step_size'],
                gamma=sched_cfg['gamma']
            )
        elif sched_type == 'CosineAnnealingWarmRestarts':
            self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=sched_cfg.get('T_0', 2000),
                T_mult=sched_cfg.get('T_mult', 2),
                eta_min=sched_cfg.get('eta_min', 1e-7)
            )
            self.scheduler_nut = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer_nut,
                T_0=sched_cfg.get('T_0', 2000),
                T_mult=sched_cfg.get('T_mult', 2),
                eta_min=sched_cfg.get('eta_min', 1e-7)
            )
        elif sched_type == 'CosineAnnealingLR':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['training']['epochs'],
                eta_min=sched_cfg.get('eta_min', 1e-7)
            )
            self.scheduler_nut = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer_nut,
                T_max=self.config['training']['epochs'],
                eta_min=sched_cfg.get('eta_min', 1e-7)
            )
        else:
            raise ValueError(f"Unknown scheduler type: {sched_type}")

        n_flow = sum(p.numel() for p in flow_params if p.requires_grad)
        n_nut = sum(p.numel() for p in nut_params if p.requires_grad)
        print(f"Optimizer (flow): AdamW — {n_flow:,} parameters")
        print(f"Optimizer (nut):  AdamW — {n_nut:,} parameters")
        print(f"Scheduler: {sched_type}")

        # AMP (Automatic Mixed Precision)
        self.use_amp = self.config['training'].get('use_amp', False)
        if self.use_amp:
            self.scaler = torch.amp.GradScaler('cuda')
            print("Mixed precision (AMP): ENABLED")
        else:
            self.scaler = None
            print("Mixed precision (AMP): disabled")

    def compute_total_loss(self, phase='diastolic'):
        """
        Compute total loss for a given phase.

        Args:
            phase: 'systolic' or 'diastolic'

        Returns:
            total_loss: Total weighted loss
            loss_dict: Dictionary of individual losses (scalars)
            wss_pred: Predicted WSS tensor
            individual_losses: Dict of live loss tensors (for adaptive weights)
        """
        tensors = self.data[phase]
        weights = self.config['loss_weights']

        # WSS Loss (CRITICAL) - Process in batches to save memory
        # Non-dimensional: both prediction and target in viscous-scaled units
        wall_batch_size = self.config['training'].get('wall_batch_size', 2000)
        n_wall_points = tensors['x'].shape[0]

        loss_wss_total = 0.0
        wss_pred_list = []

        for i in range(0, n_wall_points, wall_batch_size):
            end_idx = min(i + wall_batch_size, n_wall_points)

            loss_wss_batch, wss_pred_batch = compute_wss_loss(
                self.networks['u'], self.networks['v'], self.networks['w'],
                tensors['x'][i:end_idx], tensors['y'][i:end_idx], tensors['z'][i:end_idx],
                tensors['phase'][i:end_idx],
                tensors['wss_x'][i:end_idx], tensors['wss_y'][i:end_idx], tensors['wss_z'][i:end_idx],
                tensors['normals'][i:end_idx],
                coord_scale=self.ref_scales.get('coord_scale', 1.0),
                wss_std=self.wss_std,
                non_newtonian=self.non_newtonian,
                U_ref=self.ref_scales['U_ref'],
                L_ref=self.config['data']['normalization']['length_scale'],
            )

            loss_wss_total += loss_wss_batch * (end_idx - i) / n_wall_points
            wss_pred_list.append(wss_pred_batch.detach())

        loss_wss = loss_wss_total
        wss_pred = torch.cat(wss_pred_list, dim=0)

        # Physics Loss (Non-dimensional N-S) - Process in batches
        physics_cfg = self.config['physics']
        interior_batch_size = physics_cfg.get('interior_batch_size', 500)
        n_interior_points = tensors['x_interior'].shape[0]

        loss_physics_total = 0.0
        residual_sums = {
            'momentum_x': 0.0, 'momentum_y': 0.0, 'momentum_z': 0.0, 'continuity': 0.0,
            'nut_mean': 0.0,
        }
        nut_max_val = 0.0
        mu_ratio_mean_sum = 0.0
        mu_ratio_max_val = 1.0

        for i in range(0, n_interior_points, interior_batch_size):
            end_idx = min(i + interior_batch_size, n_interior_points)

            loss_physics_batch, residuals_batch = compute_physics_loss(
                self.networks['u'], self.networks['v'], self.networks['w'], self.networks['p'],
                self.networks['nut'],
                tensors['x_interior'][i:end_idx], tensors['y_interior'][i:end_idx],
                tensors['z_interior'][i:end_idx], tensors['phase_interior'][i:end_idx],
                Re=self.Re,
                coord_scale=self.ref_scales.get('coord_scale', 1.0),
                pressure_std=self.ref_scales.get('pressure_std', 1.0),
                non_newtonian=self.non_newtonian,
                U_ref=self.ref_scales['U_ref'],
                L_ref=self.config['data']['normalization']['length_scale'],
            )

            loss_physics_total += loss_physics_batch * (end_idx - i) / n_interior_points
            batch_weight = (end_idx - i) / n_interior_points

            for key in ('momentum_x', 'momentum_y', 'momentum_z', 'continuity', 'nut_mean'):
                if key in residuals_batch:
                    residual_sums[key] += residuals_batch[key] * batch_weight

            nut_max_val = max(nut_max_val, residuals_batch.get('nut_max', 0.0))
            mu_ratio_mean_sum += residuals_batch.get('mu_ratio_mean', 1.0) * batch_weight
            mu_ratio_max_val = max(mu_ratio_max_val, residuals_batch.get('mu_ratio_max', 1.0))

        loss_physics = loss_physics_total
        residuals = residual_sums

        # No-slip BC Loss - Process in batches
        loss_bc_total = 0.0
        for i in range(0, n_wall_points, wall_batch_size):
            end_idx = min(i + wall_batch_size, n_wall_points)

            loss_bc_batch = compute_noslip_loss(
                self.networks['u'], self.networks['v'], self.networks['w'],
                tensors['x'][i:end_idx], tensors['y'][i:end_idx], tensors['z'][i:end_idx],
                tensors['phase'][i:end_idx]
            )
            loss_bc_total += loss_bc_batch * (end_idx - i) / n_wall_points

        loss_bc = loss_bc_total

        # Pressure Loss - Process in batches
        loss_pressure_total = 0.0
        for i in range(0, n_wall_points, wall_batch_size):
            end_idx = min(i + wall_batch_size, n_wall_points)

            loss_pressure_batch = compute_pressure_loss(
                self.networks['p'],
                tensors['x'][i:end_idx], tensors['y'][i:end_idx], tensors['z'][i:end_idx],
                tensors['phase'][i:end_idx],
                tensors['pressure'][i:end_idx]
            )
            loss_pressure_total += loss_pressure_batch * (end_idx - i) / n_wall_points

        loss_pressure = loss_pressure_total

        # ── Inlet / Outlet BC Losses ──────────────────────────────────────
        loss_inlet = torch.tensor(0.0, device=self.device)
        loss_outlet = torch.tensor(0.0, device=self.device)

        if self.io_geometry is not None:
            U_ref = self.ref_scales['U_ref']
            axial_dim = self.io_geometry['axial_dim']

            # Inlet velocity BC
            if f'x_inlet' in tensors:
                if phase == 'systolic':
                    v_inlet_nondim = self.inlet_vel_systolic / U_ref
                else:
                    v_inlet_nondim = self.inlet_vel_diastolic / U_ref

                loss_inlet = compute_inlet_velocity_loss(
                    self.networks['u'], self.networks['v'], self.networks['w'],
                    tensors['x_inlet'], tensors['y_inlet'], tensors['z_inlet'],
                    tensors['phase_inlet'],
                    u_inlet_nondim=v_inlet_nondim,
                    axial_dim=axial_dim,
                )

            # Outlet pressure BC (p = 0 → p_std = 0)
            if f'x_outlet' in tensors:
                loss_outlet = compute_outlet_pressure_loss(
                    self.networks['p'],
                    tensors['x_outlet'], tensors['y_outlet'], tensors['z_outlet'],
                    tensors['phase_outlet'],
                )

        # ── ν_t regularization (prevents collapse to zero) ─────────────────
        # Point-wise lower-bound penalty: penalises each collocation point
        # where ν_t falls below the target floor, providing N individual
        # gradient signals rather than one aggregated signal.
        loss_nut_reg = torch.tensor(0.0, device=self.device)
        nut_cfg = self.config['model'].get('nut', {})
        nut_reg_weight = nut_cfg.get('reg_weight', 1.0)
        nut_reg_target = nut_cfg.get('reg_target', 0.01)
        if nut_reg_weight > 0:
            net_in_int = torch.cat([
                tensors['x_interior'], tensors['y_interior'],
                tensors['z_interior'], tensors['phase_interior'],
            ], dim=1)
            nut_pred = self.networks['nut'](net_in_int).view(-1)
            shortfall = torch.relu(nut_reg_target - nut_pred)      # (N,) point-wise
            loss_nut_reg = nut_reg_weight * shortfall.pow(2).mean()

        # Keep live tensors for adaptive weight computation
        individual_losses = {
            'wss': loss_wss,
            'physics': loss_physics,
            'bc_noslip': loss_bc,
            'pressure': loss_pressure,
            'inlet': loss_inlet,
            'outlet': loss_outlet,
            'nut_reg': loss_nut_reg,
        }

        lambda_inlet = weights.get('lambda_inlet', 10.0)
        lambda_outlet = weights.get('lambda_outlet', 10.0)

        if self.optimizer_strategy == 'config':
            # ConFIG handles gradient balancing natively — use unit weights.
            # The total_loss here is for logging only; backprop uses per-group
            # gradients combined by the ConFIG operator in _train_epoch_config().
            total_loss = (loss_wss + loss_physics + loss_bc
                          + loss_pressure + loss_inlet + loss_outlet
                          + loss_nut_reg)
        elif self.adaptive_weights_enabled and self.adaptive_weights:
            aw_physics = self.adaptive_weights['physics']
            if self.physics_ramp_epochs > 0 and self.epoch > 0:
                ramp_factor = min(1.0, self.epoch / self.physics_ramp_epochs)
                aw_physics *= ramp_factor
            total_loss = (
                self.adaptive_weights['wss'] * loss_wss +
                aw_physics * loss_physics +
                self.adaptive_weights['bc_noslip'] * loss_bc +
                self.adaptive_weights['pressure'] * loss_pressure +
                self.adaptive_weights['inlet'] * loss_inlet +
                self.adaptive_weights['outlet'] * loss_outlet +
                self.adaptive_weights['nut_reg'] * loss_nut_reg
            )
        else:
            lambda_physics = weights['lambda_physics']
            if self.physics_ramp_epochs > 0 and self.epoch > 0:
                ramp_factor = min(1.0, self.epoch / self.physics_ramp_epochs)
                lambda_physics *= ramp_factor
            total_loss = (
                weights['lambda_WSS'] * loss_wss +
                lambda_physics * loss_physics +
                weights['lambda_BC_noslip'] * loss_bc +
                weights['lambda_pressure'] * loss_pressure +
                lambda_inlet * loss_inlet +
                lambda_outlet * loss_outlet +
                loss_nut_reg
            )

        # Loss dictionary for logging
        loss_dict = {
            'total': total_loss.item(),
            'wss': loss_wss.item(),
            'physics': loss_physics.item(),
            'bc_noslip': loss_bc.item(),
            'pressure': loss_pressure.item(),
            'inlet': loss_inlet.item(),
            'outlet': loss_outlet.item(),
            'nut_reg': loss_nut_reg.item(),
            'residual_momentum_x': residuals['momentum_x'],
            'residual_momentum_y': residuals['momentum_y'],
            'residual_momentum_z': residuals['momentum_z'],
            'residual_continuity': residuals['continuity'],
            'nut_mean': residuals.get('nut_mean', 0.0),
            'nut_max': nut_max_val,
            'mu_ratio_mean': mu_ratio_mean_sum,
            'mu_ratio_max': mu_ratio_max_val,
        }

        return total_loss, loss_dict, wss_pred, individual_losses

    def _compute_adaptive_weights(self, individual_losses):
        """Compute adaptive loss weights via gradient-norm balancing (Wang et al. 2021).

        For each loss term L_i, compute the mean absolute gradient norm over all
        network parameters.  The reference loss (default: WSS) provides the
        max gradient norm.  The adaptive weight is:
            lambda_hat_i = max|grad L_ref| / mean|grad L_i|
        Smoothed with exponential moving average (EMA).

        Args:
            individual_losses: dict of live loss tensors {name: tensor}
        """
        all_params = [p for net in self.networks.values()
                      for p in net.parameters() if p.requires_grad]

        grad_norms = {}
        for name, loss_val in individual_losses.items():
            grads = torch.autograd.grad(
                loss_val, all_params, retain_graph=True, allow_unused=True)
            total = 0.0
            count = 0
            for g in grads:
                if g is not None:
                    total += g.abs().mean().item()
                    count += 1
            grad_norms[name] = total / max(count, 1)

        self._last_grad_norms = dict(grad_norms)

        ref_name = self.adaptive_weights_ref_loss
        ref_max = grad_norms.get(ref_name, 1.0)

        alpha = self.adaptive_weights_alpha
        for name, gn in grad_norms.items():
            raw_weight = ref_max / max(gn, 1e-12)
            if name in self.adaptive_weights:
                self.adaptive_weights[name] = (
                    (1.0 - alpha) * self.adaptive_weights[name] + alpha * raw_weight)
            else:
                self.adaptive_weights[name] = raw_weight

        # Enforce floor on physics weight
        if self.physics_weight_floor > 0 and 'physics' in self.adaptive_weights:
            self.adaptive_weights['physics'] = max(
                self.adaptive_weights['physics'], self.physics_weight_floor)

        # Cap all weights to prevent over-weighting already-satisfied losses
        aw_cap = self.config.get('adaptive_weights', {}).get('weight_cap', 20.0)
        if aw_cap > 0:
            for name in self.adaptive_weights:
                self.adaptive_weights[name] = min(
                    self.adaptive_weights[name], aw_cap)

    def _compute_physics_only_loss(self, phase):
        """Compute physics loss + ν_t regularization for the nut update step.

        The regularization term penalizes ν_t values below a floor
        (nut_reg_target), breaking the stable equilibrium at ν_t = 0
        that the alternating optimization creates.
        """
        tensors = self.data[phase]
        physics_cfg = self.config['physics']
        interior_batch_size = physics_cfg.get('interior_batch_size', 500)
        n_interior = tensors['x_interior'].shape[0]

        loss_physics = torch.tensor(0.0, device=self.device)
        nut_vals = []
        for i in range(0, n_interior, interior_batch_size):
            j = min(i + interior_batch_size, n_interior)
            loss_batch, residuals_batch = compute_physics_loss(
                self.networks['u'], self.networks['v'],
                self.networks['w'], self.networks['p'],
                self.networks['nut'],
                tensors['x_interior'][i:j], tensors['y_interior'][i:j],
                tensors['z_interior'][i:j], tensors['phase_interior'][i:j],
                Re=self.Re,
                coord_scale=self.ref_scales.get('coord_scale', 1.0),
                pressure_std=self.ref_scales.get('pressure_std', 1.0),
                non_newtonian=self.non_newtonian,
                U_ref=self.ref_scales['U_ref'],
                L_ref=self.config['data']['normalization']['length_scale'],
            )
            loss_physics = loss_physics + loss_batch * (j - i) / n_interior

        # ν_t lower-bound regularization: point-wise penalty on each
        # collocation point where ν_t < target (same formulation as
        # compute_total_loss for gradient-signal consistency).
        nut_cfg = self.config['model'].get('nut', {})
        nut_reg_weight = nut_cfg.get('reg_weight', 1.0)
        nut_reg_target = nut_cfg.get('reg_target', 0.01)

        if nut_reg_weight > 0:
            net_in = torch.cat([
                tensors['x_interior'], tensors['y_interior'],
                tensors['z_interior'], tensors['phase_interior'],
            ], dim=1)
            nut_pred = self.networks['nut'](net_in).view(-1)
            shortfall = torch.relu(nut_reg_target - nut_pred)   # (N,) point-wise
            loss_nut_reg = nut_reg_weight * shortfall.pow(2).mean()
            return loss_physics + loss_nut_reg

        return loss_physics

    def _get_flow_grad_vector(self):
        """Collect gradient vector from all flow network parameters."""
        grads = []
        for name in ('u', 'v', 'w', 'p'):
            for p in self.networks[name].parameters():
                if p.grad is not None:
                    grads.append(p.grad.data.view(-1))
                else:
                    grads.append(torch.zeros(p.data.numel(), device=self.device))
        return torch.cat(grads)

    def _apply_flow_grad_vector(self, grad_vec):
        """Apply a gradient vector back to all flow network parameters."""
        offset = 0
        for name in ('u', 'v', 'w', 'p'):
            for p in self.networks[name].parameters():
                numel = p.data.numel()
                p.grad = grad_vec[offset:offset + numel].view_as(p.data).clone()
                offset += numel

    def _train_epoch_config(self):
        """Train one epoch using ConFIG conflict-free gradient method.

        Groups losses into three categories to capture the key gradient conflicts:
          1. WSS (data-fitting at wall)
          2. Physics (PDE constraint in interior)
          3. BCs (no-slip + pressure + inlet + outlet + nut_reg)

        Each group's gradient is computed separately, then combined via ConFIG
        into a single conflict-free update direction.
        """
        total_loss_sum = 0.0
        phase_dicts = []
        grad_clip = self.config['training']['gradient_clip']
        flow_params = [p for n, net in self.networks.items()
                       if n != 'nut' for p in net.parameters()]

        for phase in self.config['data']['phases']:
            _, loss_dict, _, indiv = self.compute_total_loss(phase)

            # ── Collect per-group gradient vectors ──────────────────────────
            loss_groups = {
                'wss': indiv['wss'],
                'physics': indiv['physics'],
                'bc': (indiv['bc_noslip'] + indiv['pressure']
                       + indiv['inlet'] + indiv['outlet'] + indiv['nut_reg']),
            }

            grad_vectors = []
            loss_values = []
            for name, loss_val in loss_groups.items():
                self.optimizer.zero_grad(set_to_none=True)
                loss_val.backward(retain_graph=True)
                gv = self._get_flow_grad_vector()
                grad_vectors.append(gv)
                loss_values.append(loss_val.item())

            # ── ConFIG: combine into conflict-free direction ────────────────
            stacked = torch.stack(grad_vectors, dim=0)  # (3, D)
            combined_grad = self.config_operator.calculate_gradient(
                stacked, losses=loss_values)

            # Apply combined gradient and step
            self.optimizer.zero_grad(set_to_none=True)
            self._apply_flow_grad_vector(combined_grad)
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(flow_params, grad_clip)
            self.optimizer.step()

            # ── Step 2: Update nut network (physics loss only) ──────────────
            self.optimizer_nut.zero_grad(set_to_none=True)
            physics_loss = self._compute_physics_only_loss(phase)
            physics_loss.backward()
            if grad_clip > 0:
                nut_params = list(self.networks['nut'].parameters())
                nn.utils.clip_grad_norm_(nut_params, grad_clip)
            self.optimizer_nut.step()

            total_loss_sum += loss_dict['total']
            phase_dicts.append(loss_dict)

        avg_loss = total_loss_sum / len(self.config['data']['phases'])
        return avg_loss, phase_dicts

    def _train_epoch_adaptive(self):
        """Train one epoch using adaptive weight balancing (Wang et al. 2021)."""
        total_loss_sum = 0.0
        phase_dicts = []
        grad_clip = self.config['training']['gradient_clip']

        should_update_aw = (
            self.adaptive_weights_enabled
            and self.epoch > 0
            and self.epoch % self.adaptive_weights_update_interval == 0
        )

        for phase in self.config['data']['phases']:
            self.optimizer.zero_grad(set_to_none=True)

            total_loss, loss_dict, _, indiv = self.compute_total_loss(phase)

            if should_update_aw:
                self._compute_adaptive_weights(indiv)
                total_loss, loss_dict, _, indiv = self.compute_total_loss(phase)

            total_loss.backward()
            if grad_clip > 0:
                flow_params = [p for n, net in self.networks.items()
                               if n != 'nut' for p in net.parameters()]
                nn.utils.clip_grad_norm_(flow_params, grad_clip)
            self.optimizer.step()

            self.optimizer_nut.zero_grad(set_to_none=True)
            physics_loss = self._compute_physics_only_loss(phase)
            physics_loss.backward()
            if grad_clip > 0:
                nut_params = list(self.networks['nut'].parameters())
                nn.utils.clip_grad_norm_(nut_params, grad_clip)
            self.optimizer_nut.step()

            total_loss_sum += loss_dict['total']
            phase_dicts.append(loss_dict)

        avg_loss = total_loss_sum / len(self.config['data']['phases'])

        if should_update_aw and self.adaptive_weights:
            print(f"\n  Adaptive weights updated (epoch {self.epoch}): "
                  f"{', '.join(f'{k}={v:.4f}' for k, v in self.adaptive_weights.items())}")

        _NORM_FLOOR = 1e-4
        if self.loss_normalization and not self.loss_norms and phase_dicts:
            avg_raw = {k: np.mean([d[k] for d in phase_dicts])
                       for k in ['wss', 'physics', 'bc_noslip', 'pressure']}
            for k, v in avg_raw.items():
                self.loss_norms[k] = max(v, _NORM_FLOOR)
            print(f"\n  Loss normalization initialized: {self.loss_norms}")

        if (self.loss_normalization and self.renorm_interval > 0
                and self.loss_norms and self.epoch > 1
                and self.epoch % self.renorm_interval == 0 and phase_dicts):
            avg_raw = {k: np.mean([d[k] for d in phase_dicts])
                       for k in ['wss', 'physics', 'bc_noslip', 'pressure']}
            for k, v in avg_raw.items():
                self.loss_norms[k] = max(v, _NORM_FLOOR)
            print(f"\n  Loss norms updated (epoch {self.epoch}): {self.loss_norms}")

        return avg_loss, phase_dicts

    def train_epoch(self):
        """Train for one epoch using the configured optimizer strategy."""
        if self.optimizer_strategy == 'config':
            return self._train_epoch_config()
        else:
            return self._train_epoch_adaptive()

    def _compute_pressure_metrics(self, phase):
        """Compute pressure MAE, RMSE, R² in physical Pa units."""
        tensors = self.data[phase]
        n = tensors['x'].shape[0]
        batch = self.config['training'].get('wall_batch_size', 2000)

        p_pred_list = []
        for i in range(0, n, batch):
            j = min(i + batch, n)
            inp = torch.cat([tensors['x'][i:j], tensors['y'][i:j],
                             tensors['z'][i:j], tensors['phase'][i:j]], dim=1)
            with torch.no_grad():
                p_pred_list.append(self.networks['p'](inp).view(-1))

        p_pred_nd = torch.cat(p_pred_list)
        p_true_nd = tensors['pressure'].view(-1)

        P_ref = self.ref_scales['P_ref']
        pressure_std = self.ref_scales.get('pressure_std', 1.0)
        p_pred_pa = p_pred_nd * pressure_std * P_ref
        p_true_pa = p_true_nd * pressure_std * P_ref

        diff = p_pred_pa - p_true_pa
        mae = torch.abs(diff).mean().item()
        rmse = torch.sqrt((diff ** 2).mean()).item()

        ss_res = (diff ** 2).sum()
        ss_tot = ((p_true_pa - p_true_pa.mean()) ** 2).sum()
        r2 = (1.0 - ss_res / (ss_tot + 1e-12)).item()

        rel_err = (torch.norm(diff) / (torch.norm(p_true_pa) + 1e-12)).item() * 100.0

        return {'pressure_mae': mae, 'pressure_rmse': rmse,
                'pressure_r2': r2, 'pressure_rel_error_pct': rel_err}

    @staticmethod
    def _r2_from_tensors(pred, true):
        """Coefficient of determination (R²) between two flat tensors."""
        ss_res = ((true - pred) ** 2).sum()
        ss_tot = ((true - true.mean()) ** 2).sum()
        return (1.0 - ss_res / (ss_tot + 1e-12)).item()

    def evaluate(self):
        """Evaluate on training data and return detailed metrics per phase.

        Returns:
            List of dicts, one per phase, each containing losses and WSS metrics.
        """
        print("\n" + "=" * 70)
        print(f"EVALUATION - Epoch {self.epoch}")
        print("=" * 70)

        results = []
        for phase in self.config['data']['phases']:
            print(f"\n{phase.upper()}:")

            # Compute losses (gradients needed for WSS)
            _, loss_dict, wss_pred, _ = self.compute_total_loss(phase)

            # Print losses
            print(f"  Losses:")
            print(f"    Total:    {loss_dict['total']:.6f}")
            print(f"    WSS:      {loss_dict['wss']:.6f}")
            print(f"    Physics:  {loss_dict['physics']:.6f}")
            print(f"    BC:       {loss_dict['bc_noslip']:.6f}")
            print(f"    Pressure: {loss_dict['pressure']:.6f}")
            print(f"    Inlet:    {loss_dict['inlet']:.6f}")
            print(f"    Outlet:   {loss_dict['outlet']:.6f}")

            # Compute WSS metrics (non-dimensional)
            tensors = self.data[phase]
            wss_true = torch.cat([tensors['wss_x'], tensors['wss_y'], tensors['wss_z']], dim=1)
            metrics = compute_wss_metrics(wss_pred, wss_true)
            wss_r2 = self._r2_from_tensors(wss_pred.flatten(), wss_true.flatten())

            print(f"  WSS Metrics:")
            print(f"    Relative L2: {metrics['relative_l2']:.4f}")
            print(f"    Correlation: {metrics['correlation']:.4f}")
            print(f"    R²:          {wss_r2:.4f}")
            print(f"    MAE:         {metrics['mae']:.6f}")
            print(f"    RMSE:        {metrics['rmse']:.6f}")

            # Pressure metrics in physical Pa
            p_metrics = self._compute_pressure_metrics(phase)
            print(f"  Pressure Metrics (Pa):")
            print(f"    MAE:         {p_metrics['pressure_mae']:.4f}")
            print(f"    RMSE:        {p_metrics['pressure_rmse']:.4f}")
            print(f"    R²:          {p_metrics['pressure_r2']:.4f}")
            print(f"    Rel. Error:  {p_metrics['pressure_rel_error_pct']:.2f}%")

            print(f"  Physics Residuals:")
            print(f"    Momentum X: {loss_dict['residual_momentum_x']:.6f}")
            print(f"    Momentum Y: {loss_dict['residual_momentum_y']:.6f}")
            print(f"    Momentum Z: {loss_dict['residual_momentum_z']:.6f}")
            print(f"    Continuity: {loss_dict['residual_continuity']:.6f}")
            print(f"  Turbulent Viscosity (nu_t_bar):")
            print(f"    Mean: {loss_dict['nut_mean']:.6f}")
            print(f"    Max:  {loss_dict['nut_max']:.6f}")
            if self.non_newtonian is not None:
                print(f"  Carreau-Yasuda mu_ratio (mu/mu_inf):")
                print(f"    Mean: {loss_dict.get('mu_ratio_mean', 1.0):.4f}")
                print(f"    Max:  {loss_dict.get('mu_ratio_max', 1.0):.4f}")

            results.append({
                "phase": phase,
                "epoch": self.epoch,
                "dataset": "train",
                "loss_total": loss_dict['total'],
                "loss_wss": loss_dict['wss'],
                "loss_physics": loss_dict['physics'],
                "loss_bc_noslip": loss_dict['bc_noslip'],
                "loss_pressure": loss_dict['pressure'],
                "loss_inlet": loss_dict['inlet'],
                "loss_outlet": loss_dict['outlet'],
                "wss_relative_l2": metrics['relative_l2'],
                "wss_correlation": metrics['correlation'],
                "wss_r2": wss_r2,
                "wss_mae": metrics['mae'],
                "wss_rmse": metrics['rmse'],
                "pressure_mae_pa": p_metrics['pressure_mae'],
                "pressure_rmse_pa": p_metrics['pressure_rmse'],
                "pressure_r2": p_metrics['pressure_r2'],
                "pressure_rel_error_pct": p_metrics['pressure_rel_error_pct'],
                "residual_momentum_x": loss_dict['residual_momentum_x'],
                "residual_momentum_y": loss_dict['residual_momentum_y'],
                "residual_momentum_z": loss_dict['residual_momentum_z'],
                "residual_continuity": loss_dict['residual_continuity'],
            })

        return results

    def save_checkpoint(self, filename='checkpoint.pt'):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': self.epoch,
            'config': self.config,
            'networks': {name: net.state_dict() for name, net in self.networks.items()},
            'optimizer': self.optimizer.state_dict(),
            'optimizer_nut': self.optimizer_nut.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'scheduler_nut': self.scheduler_nut.state_dict(),
            'best_loss': self.best_loss,
            'loss_history': self.loss_history,
            'no_improve_count': self.no_improve_count,
            'loss_norms': self.loss_norms,
            'renorm_interval': self.renorm_interval,
            'ref_scales': self.ref_scales,
            'Re': self.Re,
            'adaptive_weights': self.adaptive_weights,
        }

        filepath = self.output_dir / filename
        torch.save(checkpoint, filepath)
        print(f"\nCheckpoint saved: {filepath}")

    def load_checkpoint(self, checkpoint_path):
        """Load training checkpoint and resume state.

        Supports both the current format (single 'optimizer'/'scheduler' keys)
        and the legacy format ('optimizers'/'schedulers' dicts from per-network setup).
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        for name in self.networks:
            if name in checkpoint['networks']:
                self.networks[name].load_state_dict(checkpoint['networks'][name])
            else:
                print(f"Warning: network '{name}' not in checkpoint; using fresh init.")

        if 'optimizer' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        if 'optimizer_nut' in checkpoint:
            self.optimizer_nut.load_state_dict(checkpoint['optimizer_nut'])

        if 'scheduler' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler'])
        if 'scheduler_nut' in checkpoint:
            self.scheduler_nut.load_state_dict(checkpoint['scheduler_nut'])

        self.best_loss = checkpoint.get('best_loss', float('inf'))
        self.loss_history = checkpoint.get('loss_history', [])
        self.no_improve_count = checkpoint.get('no_improve_count', 0)
        self.loss_norms = checkpoint.get('loss_norms', {})
        self.adaptive_weights = checkpoint.get('adaptive_weights', {})
        self.start_epoch = int(checkpoint['epoch']) + 1
        self.epoch = int(checkpoint['epoch'])

        print(f"Loaded checkpoint: {checkpoint_path}")
        print(f"Resuming from epoch: {self.start_epoch}")
        print(f"Best loss so far: {self.best_loss:.6f}")
        if self.adaptive_weights:
            print(f"Adaptive weights restored: "
                  f"{', '.join(f'{k}={v:.4f}' for k, v in self.adaptive_weights.items())}")

    def _save_evaluation_metrics(self, eval_results):
        """Save evaluation metrics to CSV.

        Args:
            eval_results: List of dicts from evaluate(), one per phase.
        """
        if not eval_results:
            return

        csv_path = self.output_dir / "evaluation_metrics.csv"
        keys = list(eval_results[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv_mod.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(eval_results)

        print(f"Evaluation metrics saved: {csv_path}")

    def _save_loss_history(self):
        """Save loss_history.csv and loss_curves.png to output directory."""
        if not self.loss_history:
            return

        # --- CSV ---
        csv_path = self.output_dir / "loss_history.csv"
        # Collect ALL keys that appear in any history entry.
        all_keys = dict.fromkeys(
            k for entry in self.loss_history for k in entry
        )
        keys = list(all_keys)
        with open(csv_path, "w", newline="") as f:
            writer = csv_mod.DictWriter(f, fieldnames=keys, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(self.loss_history)

        # --- PNG 1: 2x1 (Total Loss, Component Losses) ---
        epochs = [r["epoch"] for r in self.loss_history]
        fig1, axes1 = plt.subplots(2, 1, figsize=(10, 10), constrained_layout=True)
        
        # Total loss
        axes1[0].semilogy(epochs, [r["total"] for r in self.loss_history])
        axes1[0].set_title("Total Loss")
        axes1[0].set_xlabel("Epoch")
        axes1[0].set_ylabel("Loss")
        axes1[0].grid(True, alpha=0.3)

        # Component losses
        for key, lbl in [("wss", "WSS"), ("physics", "Physics"),
                          ("bc_noslip", "No-Slip BC"), ("pressure", "Pressure"),
                          ("inlet", "Inlet BC"), ("outlet", "Outlet BC")]:
            vals = [r.get(key, 0.0) for r in self.loss_history]
            if any(v > 0 for v in vals):
                axes1[1].semilogy(epochs, vals, label=lbl)
        axes1[1].set_title("Component Losses")
        axes1[1].set_xlabel("Epoch")
        axes1[1].set_ylabel("Loss")
        axes1[1].legend()
        axes1[1].grid(True, alpha=0.3)
        
        fig1.suptitle(f"{self.config['experiment']['name']} — Loss Curves", fontsize=14)
        fig1.savefig(self.output_dir / "loss_curves_2x1.png", dpi=300)
        plt.close(fig1)

        # --- PNG 2: 1x2 (Physics Residuals, Learning Rate/Adaptive Weights) ---
        fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
        
        # Physics residuals
        for key, lbl in [("res_mom_x", "Mom-X"), ("res_mom_y", "Mom-Y"),
                          ("res_mom_z", "Mom-Z"), ("res_cont", "Continuity")]:
            axes2[0].semilogy(epochs, [r[key] for r in self.loss_history], label=lbl)
        axes2[0].set_title("Physics Residuals")
        axes2[0].set_xlabel("Epoch")
        axes2[0].set_ylabel("Residual")
        axes2[0].legend()
        axes2[0].grid(True, alpha=0.3)

        # Learning rate
        axes2[1].semilogy(epochs, [r["lr"] for r in self.loss_history])
        axes2[1].set_title("Learning Rate")
        axes2[1].set_xlabel("Epoch")
        axes2[1].set_ylabel("LR")
        axes2[1].grid(True, alpha=0.3)
        
        fig2.suptitle(f"{self.config['experiment']['name']} — Training Dynamics", fontsize=14)
        fig2.savefig(self.output_dir / "loss_curves_1x2.png", dpi=300)
        plt.close(fig2)

        # --- PNG 3: 1x2 (Gradient Norms, Adaptive Weights) ---
        has_aw = any("aw_wss" in r for r in self.loss_history)
        if has_aw:
            fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
            
            for name, lbl in [("wss", "WSS"), ("physics", "Physics"),
                               ("bc_noslip", "No-Slip BC"), ("pressure", "Pressure"),
                               ("inlet", "Inlet BC"), ("outlet", "Outlet BC")]:
                vals = [r.get(f"grad_norm_{name}", float('nan')) for r in self.loss_history]
                valid = [(e, v) for e, v in zip(epochs, vals) if v == v]
                if valid:
                    ep, vv = zip(*valid)
                    axes3[0].semilogy(ep, vv, label=lbl, marker='.', markersize=2, linewidth=0.8)
            axes3[0].set_title("Gradient Norms (per loss)")
            axes3[0].set_xlabel("Epoch")
            axes3[0].set_ylabel("Mean |grad|")
            axes3[0].legend()
            axes3[0].grid(True, alpha=0.3)

            for name, lbl in [("wss", "WSS"), ("physics", "Physics"),
                               ("bc_noslip", "No-Slip BC"), ("pressure", "Pressure"),
                               ("inlet", "Inlet BC"), ("outlet", "Outlet BC")]:
                vals = [r.get(f"aw_{name}", float('nan')) for r in self.loss_history]
                valid = [(e, v) for e, v in zip(epochs, vals) if v == v]
                if valid:
                    ep, vv = zip(*valid)
                    axes3[1].semilogy(ep, vv, label=lbl, marker='.', markersize=2, linewidth=0.8)
            axes3[1].set_title("Adaptive Weights")
            axes3[1].set_xlabel("Epoch")
            axes3[1].set_ylabel("Weight")
            axes3[1].legend()
            axes3[1].grid(True, alpha=0.3)
            
            fig3.suptitle(f"{self.config['experiment']['name']} — Adaptive Constraints", fontsize=14)
            fig3.savefig(self.output_dir / "loss_curves_adaptive_1x2.png", dpi=300)
            plt.close(fig3)

        print(f"Loss history saved: {csv_path}")
        print(f"Loss curves saved:  {self.output_dir / 'loss_curves_2x1.png'}, 1x2, and adaptive.")

    def train(self):
        """Main training loop.

        PINNs training: all CFD data is used for supervision (no train/val
        split).  The physics loss on collocation points acts as the
        regularizer.  Early stopping tracks the total training loss.
        """
        print("\n" + "=" * 70)
        print("STARTING TRAINING")
        print("=" * 70)

        total_epochs = self.config['training']['epochs']
        eval_interval = self.config['training'].get('eval_interval', 500)
        save_interval = self.config['training']['save_interval']
        print(f"Total epochs: {total_epochs}")
        print(f"Evaluate interval: {eval_interval}")
        print(f"Save interval: {save_interval}")
        if self.early_stopping_enabled:
            print(f"Early stopping: patience={self.early_stopping_patience}, "
                  f"min_delta={self.early_stopping_min_delta}")

        start_time = time.time()

        pbar = tqdm(range(self.start_epoch, total_epochs + 1), desc="Training", unit="epoch",
                    dynamic_ncols=True, file=sys.stderr)

        for epoch in pbar:
            self.epoch = epoch

            # Dynamic collocation resampling
            if (self.resample_collocation_interval > 0
                    and epoch > 1
                    and epoch % self.resample_collocation_interval == 0):
                self._resample_collocation_points()
                print(f"\n  Collocation points resampled (epoch {epoch})")

            # Train one epoch (returns cached loss dicts — no redundant forward passes)
            avg_loss, phase_dicts = self.train_epoch()

            # Update learning rate schedulers
            self.scheduler.step()
            self.scheduler_nut.step()

            lr = self.optimizer.param_groups[0]['lr']

            # Record loss history
            avg_dict = {k: np.mean([d[k] for d in phase_dicts])
                        for k in phase_dicts[0]}
            history_entry = {
                "epoch": epoch,
                "total": avg_dict["total"],
                "wss": avg_dict["wss"],
                "physics": avg_dict["physics"],
                "bc_noslip": avg_dict["bc_noslip"],
                "pressure": avg_dict["pressure"],
                "inlet": avg_dict.get("inlet", 0.0),
                "outlet": avg_dict.get("outlet", 0.0),
                "res_mom_x": avg_dict["residual_momentum_x"],
                "res_mom_y": avg_dict["residual_momentum_y"],
                "res_mom_z": avg_dict["residual_momentum_z"],
                "res_cont": avg_dict["residual_continuity"],
                "nut_mean": avg_dict.get("nut_mean", 0.0),
                "nut_max": avg_dict.get("nut_max", 0.0),
                "mu_ratio_mean": avg_dict.get("mu_ratio_mean", 1.0),
                "mu_ratio_max": avg_dict.get("mu_ratio_max", 1.0),
                "lr": lr,
            }

            # Gradient norms and adaptive weights (populated on update epochs)
            if self.adaptive_weights_enabled:
                for name in ['wss', 'physics', 'bc_noslip', 'pressure', 'inlet', 'outlet']:
                    history_entry[f"grad_norm_{name}"] = self._last_grad_norms.get(name, float('nan'))
                    history_entry[f"aw_{name}"] = self.adaptive_weights.get(name, float('nan'))

            self.loss_history.append(history_entry)

            # Update tqdm bar
            pbar.set_postfix(loss=f"{avg_loss:.4g}", lr=f"{lr:.2g}")

            # Detailed evaluation (metrics, residuals)
            if epoch % eval_interval == 0:
                self.evaluate()

            # Persist loss history periodically
            if epoch % save_interval == 0:
                self._save_loss_history()

            # Best model tracking on training loss (PINNs — no data split)
            improved = avg_loss < (self.best_loss - self.early_stopping_min_delta)
            if improved:
                self.best_loss = avg_loss
                self.no_improve_count = 0
                self.save_checkpoint('best_model.pt')
            else:
                self.no_improve_count += 1

            if (
                self.early_stopping_enabled
                and self.no_improve_count >= self.early_stopping_patience
            ):
                print(f"\nEarly stopping triggered at epoch {epoch}.")
                print(f"No improvement for {self.no_improve_count} epochs "
                      f"(patience={self.early_stopping_patience}, "
                      f"min_delta={self.early_stopping_min_delta}).")
                break

        # Final evaluation
        print("\n" + "=" * 70)
        print("TRAINING COMPLETE")
        print("=" * 70)
        eval_results = self.evaluate()

        # Save loss curves and evaluation metrics (best_model.pt already saved during training)
        self._save_loss_history()
        self._save_evaluation_metrics(eval_results)

        # Generate CFD vs PINN comparison plots
        geom = self.config['data']['geometry']
        best_ckpt = self.output_dir / 'best_model.pt'
        if best_ckpt.exists():
            print("\nGenerating comparison plots...")
            generate_comparison_plots(geom, str(best_ckpt), self.device)

        total_time = time.time() - start_time
        print(f"\nTotal training time: {total_time/3600:.2f} hours")
        print(f"Best loss: {self.best_loss:.6f}")


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(description='Train TAA-PINN for single geometry')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration YAML file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')

    args = parser.parse_args()

    # Create trainer and run training
    trainer = TAATrainer(args.config, resume_checkpoint=args.resume)
    trainer.train()


if __name__ == "__main__":
    main()
