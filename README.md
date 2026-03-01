# Physics-Informed Neural Networks for Hemodynamic Prediction in Thoracic Aortic Aneurysms

> **A RANS-Constrained Sparse Data Approach**

This repository contains the source code accompanying the paper:

> *Physics-Informed Neural Networks for Hemodynamic Prediction in Thoracic Aortic Aneurysms: A RANS-Constrained Sparse Data Approach*

## Abstract

We present a physics-informed neural network (PINN) framework for predicting hemodynamic fields in thoracic aortic aneurysms (TAAs). The model learns from sparse CFD wall data — using approximately one-third of available measurements — whilst enforcing Reynolds-averaged Navier–Stokes (RANS) equations with a learnable turbulent viscosity field. The framework incorporates non-Newtonian Carreau–Yasuda blood rheology, pulsatile inlet conditions, and a gradient-norm adaptive loss weighting scheme that dynamically balances data-driven and physics-based objectives. An alternating dual-optimiser strategy ensures the turbulent viscosity network receives dedicated gradient signal from the governing equations. Evaluated across six anatomically distinct aneurysm configurations spanning two diameters and three morphologies, the framework achieves a mean WSS Pearson correlation of 0.929 and mean absolute error of 0.174 Pa.

## Repository Structure

```
TAA-aneurysm/
├── data/                       # CFD simulation data (CSV files)
├── src/
│   ├── data/loader.py          # Data loading and non-dimensionalisation
│   ├── models/
│   │   ├── networks.py         # Network architecture (Fourier + residual MLP)
│   │   ├── blocks.py           # Residual blocks, Swish activation
│   │   └── fourier.py          # Fourier feature encoding
│   ├── losses/
│   │   ├── wss.py              # WSS loss computation
│   │   ├── physics.py          # RANS residual loss
│   │   └── boundary.py         # No-slip and pressure boundary losses
│   ├── training/trainer.py     # Training loop with adaptive loss weighting
│   └── utils/
│       ├── geometry.py         # Wall normals, collocation point sampling
│       └── plotting.py         # Visualisation utilities
├── experiments/                # Training outputs (metrics, figures)
├── train.py                    # Entry point
└── requirements.txt            # Python dependencies
```

## Setup

```bash
git clone https://github.com/michaelajao/TAA-aneurysm.git
cd TAA-aneurysm

conda create -n taa_pinn python=3.10 -y
conda activate taa_pinn
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

```bash
python train.py --config <path_to_config>.yaml --resume <path_to_checkpoint>.pt
```

<!-- ## Dataset

CFD wall data from six TAA geometries, each resolved at systolic and diastolic cardiac phases (12 datasets total). Simulations were performed using the SST *k*–*ω* transition model with Carreau–Yasuda rheology.

| Code | Morphology | Diameter |
|------|------------|----------|
| AS5 | Axisymmetric (fusiform) | 5 cm |
| AS6 | Axisymmetric (fusiform) | 6 cm |
| AD5 | Anterior-dominant (saccular) | 5 cm |
| AD6 | Anterior-dominant (saccular) | 6 cm |
| PD5 | Posterior-dominant (saccular) | 5 cm |
| PD6 | Posterior-dominant (saccular) | 6 cm |

Each CSV contains wall surface coordinates (*X*, *Y*, *Z*), pressure, and WSS vector components. -->

## Method Overview

The framework uses five scalar-valued neural networks sharing the input (*x*, *y*, *z*, *t*_phase): four for the flow variables (*u*, *v*, *w*, *p*) and one for the learnable turbulent viscosity *ν*_t. Each branch employs Fourier feature encoding followed by a residual MLP with Swish activations. The total loss combines WSS matching, RANS momentum and continuity residuals, no-slip boundary enforcement, and pressure supervision, balanced by gradient-norm adaptive weighting by (Wang et al., 2021). Full methodological details are provided in the accompanying paper.

## Citation

```bibtex
@article{ajao2026taa_pinn,
  title   = {Physics-Informed Neural Networks for Hemodynamic Prediction in
             Thoracic Aortic Aneurysms: A RANS-Constrained Sparse Data Approach},
  author  = {Ajao, Michael and {others}},
  journal = {},
  year    = {2026},
  note    = {Manuscript in preparation}
}
```
