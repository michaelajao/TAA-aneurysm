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
from src.losses.boundary import compute_noslip_loss, compute_pressure_loss


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

        # Save configuration to output directory
        config_save_path = self.output_dir / "config.yaml"
        with open(config_save_path, 'w') as f:
            yaml.dump(self.config, f)

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
            tensors['phase_interior'] = tensors['phase'][:self.config['physics']['n_interior_points']]

            self.data[phase] = tensors

            # Print statistics
            print(f"  Wall points: {tensors['x'].shape[0]}")
            print(f"  Interior points: {x_int.shape[0]}")
            print(f"  Normals computed: {normals.shape}")
            print(f"  Phase encoding: {tensors['phase'][0].item()}")

        print(f"\nData loading complete!")

    def _resample_collocation_points(self):
        """Re-sample interior collocation points for physics loss."""
        for phase in self.config['data']['phases']:
            tensors = self.data[phase]
            x_int, y_int, z_int = sample_interior_points_torch(
                tensors['x'], tensors['y'], tensors['z'],
                n_samples=self.config['physics']['n_interior_points'],
                offset_range=self.config['physics']['interior_offset_range'],
                normals=tensors['normals'],
                seed=None,  # different points each time
                device=self.device
            )
            tensors['x_interior'] = x_int
            tensors['y_interior'] = y_int
            tensors['z_interior'] = z_int
            tensors['phase_interior'] = tensors['phase'][:self.config['physics']['n_interior_points']]

    def _initialize_networks(self):
        """Initialize neural networks."""
        print("\n" + "-" * 70)
        print("INITIALIZING NETWORKS")
        print("-" * 70)

        self.networks = create_taa_networks(
            input_dim=self.config['model']['input_dim'],
            hidden_dim=self.config['model']['hidden_dim'],
            num_layers=self.config['model']['num_layers'],
            num_frequencies=self.config['model']['num_frequencies'],
            fourier_scale=self.config['model']['fourier_scale'],
            use_fourier=self.config['model']['use_fourier'],
            device=self.device
        )

        # Print network information
        for name, net in self.networks.items():
            n_params = count_parameters(net)
            print(f"  Net_{name}: {n_params:,} parameters")

        total_params = sum(count_parameters(net) for net in self.networks.values())
        print(f"  Total: {total_params:,} parameters")

    def _initialize_optimizers(self):
        """Initialize optimizers and schedulers."""
        print("\n" + "-" * 70)
        print("INITIALIZING OPTIMIZERS")
        print("-" * 70)

        lr = self.config['training']['learning_rate']
        print(f"Learning rate: {lr}")

        self.optimizers = {}
        self.schedulers = {}

        for name, net in self.networks.items():
            # Adam optimizer with settings from original PINN-wss
            self.optimizers[name] = optim.Adam(
                net.parameters(),
                lr=lr,
                betas=(0.9, 0.99),
                eps=1e-15
            )

            # Learning rate scheduler
            sched_cfg = self.config['training']['scheduler']
            sched_type = sched_cfg['type']
            if sched_type == 'StepLR':
                self.schedulers[name] = optim.lr_scheduler.StepLR(
                    self.optimizers[name],
                    step_size=sched_cfg['step_size'],
                    gamma=sched_cfg['gamma']
                )
            elif sched_type == 'CosineAnnealingWarmRestarts':
                self.schedulers[name] = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    self.optimizers[name],
                    T_0=sched_cfg.get('T_0', 2000),
                    T_mult=sched_cfg.get('T_mult', 2),
                    eta_min=sched_cfg.get('eta_min', 1e-7)
                )
            elif sched_type == 'CosineAnnealingLR':
                self.schedulers[name] = optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizers[name],
                    T_max=self.config['training']['epochs'],
                    eta_min=sched_cfg.get('eta_min', 1e-7)
                )
            else:
                raise ValueError(f"Unknown scheduler type: {sched_type}")

        print(f"Optimizer: Adam (β=(0.9, 0.99), ε=1e-15)")
        print(f"Scheduler: {self.config['training']['scheduler']['type']}")

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
            loss_dict: Dictionary of individual losses
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
        residual_sums = {'momentum_x': 0.0, 'momentum_y': 0.0, 'momentum_z': 0.0, 'continuity': 0.0}

        for i in range(0, n_interior_points, interior_batch_size):
            end_idx = min(i + interior_batch_size, n_interior_points)

            loss_physics_batch, residuals_batch = compute_physics_loss(
                self.networks['u'], self.networks['v'], self.networks['w'], self.networks['p'],
                tensors['x_interior'][i:end_idx], tensors['y_interior'][i:end_idx],
                tensors['z_interior'][i:end_idx], tensors['phase_interior'][i:end_idx],
                Re=self.Re,
            )

            loss_physics_total += loss_physics_batch * (end_idx - i) / n_interior_points

            for key in residual_sums:
                residual_sums[key] += residuals_batch[key] * (end_idx - i) / n_interior_points

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

        # Total weighted loss (with optional loss normalization and physics annealing)
        # Physics annealing: ramp lambda_physics from 0 to full weight over ramp epochs
        lambda_physics = weights['lambda_physics']
        if self.physics_ramp_epochs > 0 and self.epoch > 0:
            ramp_factor = min(1.0, self.epoch / self.physics_ramp_epochs)
            lambda_physics = weights['lambda_physics'] * ramp_factor

        if self.loss_normalization and self.loss_norms:
            total_loss = (
                weights['lambda_WSS'] * loss_wss / self.loss_norms.get('wss', 1.0) +
                lambda_physics * loss_physics / self.loss_norms.get('physics', 1.0) +
                weights['lambda_BC_noslip'] * loss_bc / self.loss_norms.get('bc_noslip', 1.0) +
                weights['lambda_pressure'] * loss_pressure / self.loss_norms.get('pressure', 1.0)
            )
        else:
            total_loss = (
                weights['lambda_WSS'] * loss_wss +
                lambda_physics * loss_physics +
                weights['lambda_BC_noslip'] * loss_bc +
                weights['lambda_pressure'] * loss_pressure
            )

        # Loss dictionary for logging
        loss_dict = {
            'total': total_loss.item(),
            'wss': loss_wss.item(),
            'physics': loss_physics.item(),
            'bc_noslip': loss_bc.item(),
            'pressure': loss_pressure.item(),
            'residual_momentum_x': residuals['momentum_x'],
            'residual_momentum_y': residuals['momentum_y'],
            'residual_momentum_z': residuals['momentum_z'],
            'residual_continuity': residuals['continuity']
        }

        return total_loss, loss_dict, wss_pred

    def train_epoch(self):
        """Train for one epoch. Returns (avg_loss, list_of_loss_dicts)."""
        # Train on both phases
        total_loss_sum = 0.0
        phase_dicts = []

        for phase in self.config['data']['phases']:
            # Zero gradients
            for opt in self.optimizers.values():
                opt.zero_grad(set_to_none=True)

            # Compute loss (with AMP if enabled)
            if self.use_amp:
                with torch.amp.autocast('cuda'):
                    total_loss, loss_dict, _ = self.compute_total_loss(phase)
                self.scaler.scale(total_loss).backward()
                if self.config['training']['gradient_clip'] > 0:
                    for opt in self.optimizers.values():
                        self.scaler.unscale_(opt)
                    for net in self.networks.values():
                        nn.utils.clip_grad_norm_(
                            net.parameters(),
                            self.config['training']['gradient_clip']
                        )
                for opt in self.optimizers.values():
                    self.scaler.step(opt)
                self.scaler.update()
            else:
                total_loss, loss_dict, _ = self.compute_total_loss(phase)
                total_loss.backward()
                if self.config['training']['gradient_clip'] > 0:
                    for net in self.networks.values():
                        nn.utils.clip_grad_norm_(
                            net.parameters(),
                            self.config['training']['gradient_clip']
                        )
                for opt in self.optimizers.values():
                    opt.step()

            total_loss_sum += loss_dict['total']
            phase_dicts.append(loss_dict)

        avg_loss = total_loss_sum / len(self.config['data']['phases'])

        # Initialize loss normalization on first epoch.
        # Floor of 1e-4 prevents tiny denominators (e.g. BC loss → 0 on
        # training data) from inflating validation loss by orders of magnitude.
        _NORM_FLOOR = 1e-4
        if self.loss_normalization and not self.loss_norms and phase_dicts:
            avg_raw = {k: np.mean([d[k] for d in phase_dicts])
                       for k in ['wss', 'physics', 'bc_noslip', 'pressure']}
            for k, v in avg_raw.items():
                self.loss_norms[k] = max(v, _NORM_FLOOR)
            print(f"\n  Loss normalization initialized: {self.loss_norms}")

        # Periodic renormalization: update norms to current loss magnitudes
        if (self.loss_normalization and self.renorm_interval > 0
                and self.loss_norms and self.epoch > 1
                and self.epoch % self.renorm_interval == 0 and phase_dicts):
            avg_raw = {k: np.mean([d[k] for d in phase_dicts])
                       for k in ['wss', 'physics', 'bc_noslip', 'pressure']}
            for k, v in avg_raw.items():
                self.loss_norms[k] = max(v, _NORM_FLOOR)
            print(f"\n  Loss norms updated (epoch {self.epoch}): {self.loss_norms}")

        return avg_loss, phase_dicts

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
            _, loss_dict, wss_pred = self.compute_total_loss(phase)

            # Print losses
            print(f"  Losses:")
            print(f"    Total:    {loss_dict['total']:.6f}")
            print(f"    WSS:      {loss_dict['wss']:.6f}")
            print(f"    Physics:  {loss_dict['physics']:.6f}")
            print(f"    BC:       {loss_dict['bc_noslip']:.6f}")
            print(f"    Pressure: {loss_dict['pressure']:.6f}")

            # Compute WSS metrics
            tensors = self.data[phase]
            wss_true = torch.cat([tensors['wss_x'], tensors['wss_y'], tensors['wss_z']], dim=1)
            metrics = compute_wss_metrics(wss_pred, wss_true)

            print(f"  WSS Metrics:")
            print(f"    Relative L2: {metrics['relative_l2']:.4f}")
            print(f"    Correlation: {metrics['correlation']:.4f}")
            print(f"    MAE:         {metrics['mae']:.6f}")
            print(f"    RMSE:        {metrics['rmse']:.6f}")

            print(f"  Physics Residuals:")
            print(f"    Momentum X: {loss_dict['residual_momentum_x']:.6f}")
            print(f"    Momentum Y: {loss_dict['residual_momentum_y']:.6f}")
            print(f"    Momentum Z: {loss_dict['residual_momentum_z']:.6f}")
            print(f"    Continuity: {loss_dict['residual_continuity']:.6f}")

            results.append({
                "phase": phase,
                "epoch": self.epoch,
                "dataset": "train",
                "loss_total": loss_dict['total'],
                "loss_wss": loss_dict['wss'],
                "loss_physics": loss_dict['physics'],
                "loss_bc_noslip": loss_dict['bc_noslip'],
                "loss_pressure": loss_dict['pressure'],
                "wss_relative_l2": metrics['relative_l2'],
                "wss_correlation": metrics['correlation'],
                "wss_mae": metrics['mae'],
                "wss_rmse": metrics['rmse'],
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
            'optimizers': {name: opt.state_dict() for name, opt in self.optimizers.items()},
            'schedulers': {name: sch.state_dict() for name, sch in self.schedulers.items()},
            'best_loss': self.best_loss,
            'loss_history': self.loss_history,
            'no_improve_count': self.no_improve_count,
            'loss_norms': self.loss_norms,
            'renorm_interval': self.renorm_interval,
            'ref_scales': self.ref_scales,
            'Re': self.Re,
        }

        filepath = self.output_dir / filename
        torch.save(checkpoint, filepath)
        print(f"\nCheckpoint saved: {filepath}")

    def load_checkpoint(self, checkpoint_path):
        """Load training checkpoint and resume state."""
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        for name in self.networks:
            self.networks[name].load_state_dict(checkpoint['networks'][name])

        for name in self.optimizers:
            self.optimizers[name].load_state_dict(checkpoint['optimizers'][name])

        for name in self.schedulers:
            self.schedulers[name].load_state_dict(checkpoint['schedulers'][name])

        self.best_loss = checkpoint.get('best_loss', float('inf'))
        self.loss_history = checkpoint.get('loss_history', [])
        self.no_improve_count = checkpoint.get('no_improve_count', 0)
        self.loss_norms = checkpoint.get('loss_norms', {})
        self.start_epoch = int(checkpoint['epoch']) + 1
        self.epoch = int(checkpoint['epoch'])

        print(f"Loaded checkpoint: {checkpoint_path}")
        print(f"Resuming from epoch: {self.start_epoch}")
        print(f"Best loss so far: {self.best_loss:.6f}")

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

        # --- PNG ---
        epochs = [r["epoch"] for r in self.loss_history]
        fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)

        # Total loss
        axes[0, 0].semilogy(epochs, [r["total"] for r in self.loss_history])
        axes[0, 0].set_title("Total Loss")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].grid(True, alpha=0.3)

        # Component losses
        for key, lbl in [("wss", "WSS"), ("physics", "Physics"),
                          ("bc_noslip", "No-Slip BC"), ("pressure", "Pressure")]:
            axes[0, 1].semilogy(epochs, [r[key] for r in self.loss_history], label=lbl)
        axes[0, 1].set_title("Component Losses")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("Loss")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Physics residuals
        for key, lbl in [("res_mom_x", "Mom-X"), ("res_mom_y", "Mom-Y"),
                          ("res_mom_z", "Mom-Z"), ("res_cont", "Continuity")]:
            axes[1, 0].semilogy(epochs, [r[key] for r in self.loss_history], label=lbl)
        axes[1, 0].set_title("Physics Residuals")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("Residual")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Learning rate
        axes[1, 1].semilogy(epochs, [r["lr"] for r in self.loss_history])
        axes[1, 1].set_title("Learning Rate")
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("LR")
        axes[1, 1].grid(True, alpha=0.3)

        fig.suptitle(f"{self.config['experiment']['name']} — Training Curves", fontsize=14)
        fig.savefig(self.output_dir / "loss_curves.png", dpi=200)
        plt.close(fig)
        print(f"Loss history saved: {csv_path}")
        print(f"Loss curves saved:  {self.output_dir / 'loss_curves.png'}")

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
                    dynamic_ncols=True)

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
            for scheduler in self.schedulers.values():
                scheduler.step()

            lr = self.optimizers['u'].param_groups[0]['lr']

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
                "res_mom_x": avg_dict["residual_momentum_x"],
                "res_mom_y": avg_dict["residual_momentum_y"],
                "res_mom_z": avg_dict["residual_momentum_z"],
                "res_cont": avg_dict["residual_continuity"],
                "lr": lr,
            }
            self.loss_history.append(history_entry)

            # Update tqdm bar
            pbar.set_postfix(loss=f"{avg_loss:.4e}", lr=f"{lr:.1e}")

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

        # Save final model, loss curves, and evaluation metrics
        self.save_checkpoint('final_model.pt')
        self._save_loss_history()
        self._save_evaluation_metrics(eval_results)

        # Generate CFD vs PINN comparison plots
        geom = self.config['data']['geometry']
        best_ckpt = self.output_dir / 'best_model.pt'
        if best_ckpt.exists():
            print("\nGenerating comparison plots...")
            try:
                generate_comparison_plots(geom, str(best_ckpt), self.device)
            except Exception as e:
                print(f"  Warning: comparison plot generation failed: {e}")

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
