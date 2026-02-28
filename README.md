# TAA-PINN: Physics-Informed Neural Networks for Thoracic Aortic Aneurysm Hemodynamics

## Overview

Physics-Informed Neural Networks (PINNs) for predicting hemodynamic fields in Thoracic Aortic Aneurysms (TAAs) from wall shear stress (WSS) data. Rather than requiring sparse interior velocity measurements, this approach uses **dense WSS data at the wall boundary** -- which constrains velocity gradients -- combined with Navier-Stokes physics constraints at interior collocation points to reconstruct the full 3D velocity and pressure fields.

## Key Innovation

Wall shear stress is the velocity gradient at the wall:

```
WSS = mu * (du/dn)|_wall
```

By matching WSS, the network learns the **derivative of velocity** at boundaries. Combined with Navier-Stokes equations enforced in the interior, this uniquely determines the entire flow field without requiring interior velocity measurements.

## Project Structure

```
TAA-aneurysm/
├── data/                       # CFD simulation data (CSV)
├── src/
│   ├── data/loader.py          # Data loading and non-dimensionalization
│   ├── models/
│   │   ├── networks.py         # TAANet architecture (Fourier + residual)
│   │   ├── blocks.py           # Residual blocks, Swish activation
│   │   └── fourier.py          # Fourier feature encoding
│   ├── losses/
│   │   ├── wss.py              # WSS loss and metrics
│   │   ├── physics.py          # Navier-Stokes residual loss
│   │   └── boundary.py         # No-slip BC and pressure loss
│   ├── training/trainer.py     # TAATrainer class (training loop)
│   └── utils/
│       ├── geometry.py         # Wall normals, collocation point sampling
│       └── plotting.py         # CFD vs PINN comparison plots
├── train.py                    # Entry-point training script
└── requirements.txt            # Python dependencies
```

## Setup

```bash
# Clone the repository
git clone https://github.com/michaelajao/TAA-aneurysm.git
cd TAA-aneurysm

# Create and activate a conda environment
conda create -n taa_pinn python=3.10 -y
conda activate taa_pinn

# Install dependencies
pip install -r requirements.txt
```

## Configuration

The training script expects a YAML configuration file. Create a `configs/` directory and add a config for each geometry (e.g. `configs/AS5_config.yaml`):

```yaml
data:
  geometry: AS5                   # geometry identifier
  phases: [systolic, diastolic]
  data_dir: data/
  files:
    systolic: "5cm systolic.csv"
    diastolic: "5cm diastolic.csv"
  subsample_factor: 3
  normalization:
    length_scale: 0.05

model:
  architecture: fourier_residual
  input_dim: 4
  hidden_dim: 128
  num_layers: 6
  use_fourier: true
  num_frequencies: 16
  fourier_scale: 1.0
  device: cuda                    # or "cpu"

training:
  batch_size: 4096
  wall_batch_size: 16000
  epochs: 10000
  learning_rate: 0.0001
  scheduler:
    type: CosineAnnealingLR
    eta_min: 1.0e-6
  gradient_clip: 1.0
  output_dir: experiments/AS5/    # where to save checkpoints & figures

loss_weights:
  lambda_physics: 0.01
  lambda_BC_noslip: 10.0
  lambda_WSS: 1.0
  lambda_pressure: 10.0

adaptive_weights:
  enabled: true
  update_interval: 100
  alpha: 0.9
  ref_loss: wss
```

Create one config per geometry (`AD5`, `AD6`, `AS5`, `AS6`, `PD5`, `PD6`), adjusting `data.geometry`, `data.files`, and `training.output_dir` accordingly. The data file mapping is:

| Geometry | Systolic File | Diastolic File |
|----------|---------------|----------------|
| AS5 | `5cm systolic.csv` | `5cm diastolic.csv` |
| AS6 | `6cm systolic.csv` | `6cm diastolic.csv` |
| AD5 | `5cm ASD systolic.csv` | `5cm ASD Diastolic.csv` |
| AD6 | `6cm ASD Systolic.csv` | `6cm ASD diastolic.csv` |
| PD5 | `5cm ASU systolic.csv` | `5cm ASU Diastolic.csv` |
| PD6 | `6cm ASU systolic.csv` | `6cm ASU Diastolic.csv` |

## Usage

```bash
# Train a single geometry
python -u -m src.training.trainer --config configs/AS5_config.yaml

# Resume from a checkpoint
python -u -m src.training.trainer --config configs/AS5_config.yaml \
  --resume experiments/AS5/best_model.pt

# Select a specific GPU
CUDA_VISIBLE_DEVICES=0 python -u -m src.training.trainer \
  --config configs/AS5_config.yaml

# Generate comparison plots from a trained model
python -m src.utils.plotting --geom AS5

# Generate cross-geometry summary bar chart
python -m src.utils.plotting --summary
```

## Network Architecture

Four separate networks (Net_u, Net_v, Net_w, Net_p) each predict one scalar field (velocity components u, v, w and pressure p) from spatial coordinates and cardiac phase:

```
Input (x, y, z, t_phase)  -->  4D
    |
Fourier Feature Encoding   -->  64D  (32 frequencies, scale=10.0)
    |
Linear + Swish             -->  128D
    |
10x Residual Blocks        -->  128D  (skip connections)
    |
Linear                     -->  1D
    |
Output (scalar field)
```

Each network has 338,689 parameters (1,354,756 total across all four).

## Loss Function

The total loss combines four terms:

```
L_total = lambda_WSS * L_WSS + lambda_physics * L_physics + lambda_BC * L_BC + lambda_pressure * L_pressure
```

| Loss Term | Description | Computation |
|-----------|-------------|-------------|
| **L_WSS** | Wall shear stress matching | MSE between predicted and CFD WSS at wall points. Predicted WSS is derived from velocity gradients via the stress tensor. |
| **L_physics** | Navier-Stokes residuals | MSE of momentum and continuity equation residuals at interior collocation points. |
| **L_BC** | No-slip boundary condition | MSE of predicted velocity at wall (target: zero). |
| **L_pressure** | Pressure matching | MSE between predicted and CFD pressure at wall points. |

### Adaptive Loss Weighting (Wang et al. 2021)

Instead of fixed loss weights, the trainer implements gradient-norm balancing from *"Understanding and Mitigating Gradient Flow Pathologies in Physics-Informed Neural Networks"* (Wang, Teng & Perdikaris, SIAM J. Sci. Comput., 2021):

- For each loss term, compute the mean gradient norm across all network parameters
- Set adaptive weight: `lambda_hat_i = max|grad L_ref| / mean|grad L_i|`
- Smooth with exponential moving average: `lambda_i = (1 - alpha) * lambda_i_old + alpha * lambda_hat_i`
- Update every N epochs (configurable, default: 100)

This ensures all loss terms contribute equally in gradient magnitude regardless of their raw scale, preventing any single term from dominating training.

## Non-Dimensional Formulation

All quantities are non-dimensionalized for numerical stability:

| Scale | Definition | Typical Value (AS5) |
|-------|-----------|---------------------|
| L_ref | Vessel diameter | 0.05 m |
| U_ref | sqrt(max(\|p\|) / rho) | 0.62 m/s |
| P_ref | rho * U_ref^2 | 407 Pa |
| tau_ref | mu * U_ref / L_ref | 0.043 Pa |
| Re | rho * U_ref * L_ref / mu | 9,383 |

The non-dimensional Navier-Stokes equations reduce to:

```
u_bar . grad(u_bar) = -grad(p_bar) + (1/Re) * laplacian(u_bar)
div(u_bar) = 0
```

In viscous scaling, the mu factor cancels in the WSS formulation, making the loss computation cleaner.

## Training Features

- **Adaptive loss weights**: Gradient-norm balancing (Wang et al. 2021) with EMA smoothing
- **Physics ramp**: Gradually introduces physics loss over configurable number of epochs
- **Smooth collocation resampling**: Partial replacement of interior points (configurable fraction) to avoid loss spikes
- **Gradient analysis**: Per-loss gradient norms logged for diagnostics
- **Expanded training curves**: 3x2 plot grid with gradient norms and adaptive weight history
- **Early stopping**: Patience-based on total training loss
- **Checkpoint resume**: Full state recovery including adaptive weights

## Dataset

TAA CFD data from 6 patient-specific geometries, each with systolic and diastolic phases (12 files total).

| Code | Description | Diameter | Shape (beta) | Type |
|------|-------------|----------|-----|------|
| AS5 | Standard | 5.0 cm | 1.00 | Axisymmetric (Fusiform) |
| PD5 | ASU | 5.0 cm | 1.78 | Posterior-Dominant (Saccular) |
| AD5 | ASD | 5.0 cm | 0.56 | Anterior-Dominant (Saccular) |
| AS6 | Standard | 6.0 cm | 1.00 | Axisymmetric (Fusiform) |
| PD6 | ASU | 6.0 cm | 1.61 | Posterior-Dominant (Saccular) |
| AD6 | ASD | 6.0 cm | 0.62 | Anterior-Dominant (Saccular) |

Each CSV contains wall surface coordinates (X, Y, Z), pressure, and WSS vector components from CFD simulations. Wall points are subsampled (default 10x) to ~7,085 points per phase.


<!-- ## References

- Arzani, A., Wang, J.X., & D'Souza, R.M. (2021). "Uncovering near-wall blood flow from sparse data with physics-informed neural networks." *Physics of Fluids*, 33(7).
- Wang, S., Teng, Y., & Perdikaris, P. (2021). "Understanding and mitigating gradient flow pathologies in physics-informed neural networks." *SIAM J. Sci. Comput.*, 43(5), A3055-A3081.
- Raissi, M., Perdikaris, P., & Karniadakis, G.E. (2019). "Physics-informed neural networks: A deep learning framework for solving forward and inverse problems." *J. Comput. Phys.*, 378, 686-707.
- Tancik, M., et al. (2020). "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains." *NeurIPS 2020*. -->
