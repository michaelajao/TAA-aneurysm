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

**Requirements:** PyTorch ≥ 2.0, Open3D ≥ 0.17, NumPy, Pandas, Matplotlib, SciPy, PyYAML. A CUDA-capable GPU is recommended.

## Usage

Training is configured via YAML files passed to `train.py`:

```bash
python train.py --config <path_to_config>.yaml
```

Each configuration specifies the geometry, data file paths, model hyperparameters, and loss weights. Refer to the existing source code (particularly `src/training/trainer.py` and `src/data/loader.py`) for the full set of configurable options.

To resume from a checkpoint:

```bash
python train.py --config <path_to_config>.yaml --resume <path_to_checkpoint>.pt
```

## Dataset

CFD wall data from six TAA geometries, each resolved at systolic and diastolic cardiac phases (12 datasets total). Simulations were performed using the SST *k*–*ω* transition model with Carreau–Yasuda rheology.

| Code | Morphology | Diameter |
|------|------------|----------|
| AS5 | Axisymmetric (fusiform) | 5 cm |
| AS6 | Axisymmetric (fusiform) | 6 cm |
| AD5 | Anterior-dominant (saccular) | 5 cm |
| AD6 | Anterior-dominant (saccular) | 6 cm |
| PD5 | Posterior-dominant (saccular) | 5 cm |
| PD6 | Posterior-dominant (saccular) | 6 cm |

Each CSV contains wall surface coordinates (*X*, *Y*, *Z*), pressure, and WSS vector components.

## Method Overview

The framework uses five scalar-valued neural networks sharing the input (*x*, *y*, *z*, *t*_phase): four for the flow variables (*u*, *v*, *w*, *p*) and one for the learnable turbulent viscosity *ν*_t. Each branch employs Fourier feature encoding followed by a residual MLP with Swish activations. The total loss combines WSS matching, RANS momentum and continuity residuals, no-slip boundary enforcement, and pressure supervision, balanced by gradient-norm adaptive weighting (Wang et al., 2021). Full methodological details are provided in the accompanying paper.

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

## References

- Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. *Journal of Computational Physics*, 378, 686–707.
- Wang, S., Teng, Y., & Perdikaris, P. (2021). Understanding and mitigating gradient flow pathologies in physics-informed neural networks. *SIAM Journal on Scientific Computing*, 43(5), A3055–A3081.
- Tancik, M., et al. (2020). Fourier features let networks learn high frequency functions in low dimensional domains. *NeurIPS 2020*.
