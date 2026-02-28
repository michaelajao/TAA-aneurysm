# Physics-Informed Neural Networks for Hemodynamic Prediction in Thoracic Aortic Aneurysms

> **A RANS-Constrained Sparse Data Approach with Learnable Turbulent Viscosity**

## Abstract

Predicting hemodynamic parameters within thoracic aortic aneurysms (TAAs) is critical for assessing rupture and dissection risk in cardiovascular surgery. We present a physics-informed neural network (PINN) framework that learns from sparse CFD wall data — using approximately one-third of available measurements — whilst simultaneously enforcing Reynolds-averaged Navier–Stokes (RANS) equations with a learnable turbulent viscosity field. The approach integrates a non-Newtonian Carreau–Yasuda viscosity model to capture blood's shear-thinning behaviour, and predicts wall shear stress (WSS) patterns across multiple anatomically distinct aneurysm configurations spanning two diameters and three morphologies. A gradient-norm adaptive loss weighting scheme dynamically balances data-driven and physics-based loss terms, while an alternating dual-optimiser strategy ensures the turbulent viscosity network receives dedicated gradient signal. Evaluated on five completed configurations, the framework achieves a **mean WSS Pearson correlation of 0.929** and **mean absolute error of 0.174 Pa**.

**Keywords:** Physics-Informed Neural Networks · Thoracic Aortic Aneurysms · Haemodynamics · Wall Shear Stress · RANS · Adaptive Loss Weighting · Sparse Data Learning

## Motivation

Traditional CFD simulations provide detailed haemodynamic information but demand substantial computational resources and time-consuming mesh generation for patient-specific geometries. PINNs offer an alternative by embedding physical laws directly into the learning process, enabling interpolation from sparse wall measurements while maintaining consistency with fundamental fluid mechanics. This framework demonstrates that clinically relevant WSS accuracy can be achieved from boundary-only data by enforcing conservation laws at interior collocation points.

## Method Overview

### Governing Equations

The PINN solves the incompressible RANS equations in a quasi-steady formulation at two cardiac phases (systole and diastole):

$$\nabla \cdot \mathbf{u} = 0$$

$$\rho \, \mathbf{u} \cdot \nabla \mathbf{u} = -\nabla p + \nabla \cdot \boldsymbol{\tau}_{\text{eff}}$$

where the effective stress combines a shear-dependent molecular viscosity (Carreau–Yasuda model) with a **learnable turbulent viscosity** $\nu_t(\mathbf{x}; t_{\text{phase}})$ output by a dedicated neural network branch — eliminating the need for explicit turbulence transport equations.

### Network Architecture

Four separate networks (Net_u, Net_v, Net_w, Net_p) each predict one scalar field from spatial coordinates and cardiac phase. Each uses Fourier feature encoding followed by residual blocks with Swish activation. A fifth network predicts the turbulent viscosity field.

### Multi-Objective Loss

The total loss balances four terms — WSS matching, RANS residuals, no-slip boundary conditions, and pressure matching — using gradient-norm adaptive weighting (Wang et al., 2021) with exponential moving average smoothing.

### Non-Dimensionalisation

All quantities are non-dimensionalised using characteristic scales derived from the data (vessel diameter, pressure-based velocity scale) for numerical stability.

## Dataset

CFD wall data from 6 patient-specific TAA geometries, each at systolic and diastolic cardiac phases (12 datasets total). Simulations were performed with the SST k–ω transition model and Carreau–Yasuda blood rheology.

| Code | Morphology | Diameter | β | Type |
|------|------------|----------|------|------|
| AS5 | Axisymmetric | 5.0 cm | 1.00 | Fusiform |
| AS6 | Axisymmetric | 6.0 cm | 1.00 | Fusiform |
| AD5 | Anterior-Dominant | 5.0 cm | 0.56 | Saccular |
| AD6 | Anterior-Dominant | 6.0 cm | 0.62 | Saccular |
| PD5 | Posterior-Dominant | 5.0 cm | 1.78 | Saccular |
| PD6 | Posterior-Dominant | 6.0 cm | 1.61 | Saccular |

Each CSV contains wall surface coordinates (X, Y, Z), pressure, and WSS vector components. Wall points are subsampled to ~7,085 points per phase.

## Repository Structure

```
TAA-aneurysm/
├── data/                       # CFD wall data (CSV, 12 files)
├── src/
│   ├── data/loader.py          # Data loading and non-dimensionalisation
│   ├── models/
│   │   ├── networks.py         # TAANet architecture (Fourier + residual)
│   │   ├── blocks.py           # Residual blocks, Swish activation
│   │   └── fourier.py          # Fourier feature encoding
│   ├── losses/
│   │   ├── wss.py              # WSS loss computation
│   │   ├── physics.py          # RANS residual loss
│   │   └── boundary.py         # No-slip and pressure boundary losses
│   ├── training/trainer.py     # Training loop with adaptive weighting
│   └── utils/
│       ├── geometry.py         # Wall normals, collocation sampling
│       └── plotting.py         # CFD vs PINN comparison visualisation
├── train.py                    # Entry-point script
├── experiments/                # Output: metrics, figures, checkpoints
└── requirements.txt
```

## Reproducing the Results

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended)

### Installation

```bash
git clone https://github.com/michaelajao/TAA-aneurysm.git
cd TAA-aneurysm

conda create -n taa_pinn python=3.10 -y
conda activate taa_pinn

pip install -r requirements.txt
```

### Training

Training requires a YAML configuration file specifying the geometry, data paths, model hyperparameters, and loss weights. See the paper for full hyperparameter details.

```bash
# Train on a specific geometry
python -u -m src.training.trainer --config <path-to-config.yaml>

# Resume from checkpoint
python -u -m src.training.trainer --config <path-to-config.yaml> \
  --resume experiments/<GEOM>/best_model.pt
```

## Key References

- Raissi, M., Perdikaris, P., & Karniadakis, G.E. (2019). "Physics-informed neural networks: A deep learning framework for solving forward and inverse problems." *J. Comput. Phys.*, 378, 686–707.
- Wang, S., Teng, Y., & Perdikaris, P. (2021). "Understanding and mitigating gradient flow pathologies in physics-informed neural networks." *SIAM J. Sci. Comput.*, 43(5), A3055–A3081.
- Arzani, A., Wang, J.X., & D'Souza, R.M. (2021). "Uncovering near-wall blood flow from sparse data with physics-informed neural networks." *Physics of Fluids*, 33(7).

## License

This project is part of ongoing research. Please contact the authors before reuse.

<!-- ## Citation

If you use this code in your research, please cite:

```bibtex
@article{taa_pinn_2026,
  title={Physics-Informed Neural Networks for Hemodynamic Prediction in Thoracic Aortic Aneurysms: A RANS-Constrained Sparse Data Approach},
  author={},
  year={2026}
}
```
-->
