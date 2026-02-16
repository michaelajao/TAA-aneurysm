# TAA-PINN: Physics-Informed Neural Networks for Thoracic Aortic Aneurysm Analysis

## Overview

This project implements Physics-Informed Neural Networks (PINNs) for hemodynamic analysis of Thoracic Aortic Aneurysms (TAAs) using Computational Fluid Dynamics (CFD) data. The key innovation is using **wall shear stress (WSS) data** to constrain the learning of interior velocity and pressure fields, rather than requiring sparse interior velocity measurements.

**Current Version**: Non-Dimensional Formulation with Re-Based Physics
- **Full non-dimensionalization** with proper Reynolds number formulation
- Reference scales auto-computed from data and fluid properties
- Correct blood properties: ρ = 1060 kg/m³, μ = 0.0035 Pa·s
- Physics loss annealing for stable convergence
- Periodic loss renormalization to prevent component stagnation
- Dynamic collocation point resampling for better coverage
- Validation split (20%) for generalization monitoring

## Project Status

✅ **Implemented**:
- Four-network architecture (u, v, w, p) with Fourier features
- **Proper non-dimensionalization** with Re-based formulation (Re ≈ 9,400)
- WSS-constrained training with automatic differentiation (viscous scaling)
- Navier-Stokes physics enforcement with correct blood properties
- Adaptive loss balancing and physics annealing
- Validation metrics and early stopping
- CFD vs PINN comparison visualization

⚠️ **Recent Major Improvements**:
- **V4 (Current)**: Full non-dimensionalization with proper Re formulation, correct blood properties
- **V3**: Physics loss divergence fixed (λ_physics restored to 1.0 with annealing)
- **V3**: Coordinate centering now consistent between training and evaluation
- **V3**: Loss normalization updates periodically to track changing magnitudes
- **V3**: Best model saves immediately on improvement

## Table of Contents

1. [Quick Start](#quick-start)
2. [What Gets Trained](#what-gets-trained)
3. [Training Methodology](#training-methodology)
4. [Dataset Description](#dataset-description)
5. [Installation & Usage](#installation--usage)
6. [Project Structure](#project-structure)
7. [Results & Validation](#results--validation)
8. [References](#references)

---

## Quick Start

```bash
# 1. Install dependencies
conda activate deep_tf  # or your environment
pip install -r requirements.txt

# 2. Train a single geometry
python -m src.training.trainer --config configs/AS5_config.yaml

# 3. Train all geometries
bash scripts/run_training.sh

# 4. Train a subset of geometries
GEOMS="AS5 AD5" bash scripts/run_training.sh

# 5. Generate comparison plots
python -m src.utils.plotting --geom AS5

# 6. Monitor training
tail -f logs/gpu1_*.log
```

---

## What Gets Trained

### Neural Networks (4 separate networks)

The system trains **four separate neural networks** simultaneously, each predicting one component of the fluid flow:

1. **Net_u**: Predicts **x-velocity component** (horizontal velocity)
   - Input: (x, y, z, t_phase) - spatial coordinates + cardiac phase
   - Output: u(x, y, z, t) - scalar velocity field
   - Parameters: 1,332,737

2. **Net_v**: Predicts **y-velocity component** (vertical velocity)
   - Input: (x, y, z, t_phase)
   - Output: v(x, y, z, t)
   - Parameters: 1,332,737

3. **Net_w**: Predicts **z-velocity component** (depth velocity)
   - Input: (x, y, z, t_phase)
   - Output: w(x, y, z, t)
   - Parameters: 1,332,737

4. **Net_p**: Predicts **pressure field**
   - Input: (x, y, z, t_phase)
   - Output: p(x, y, z, t)
   - Parameters: 1,332,737

**Total Parameters**: 5,330,948 trainable parameters

### Network Architecture (per network)

```
Input (4D: x,y,z,t)
    ↓
Fourier Feature Encoding (32 frequencies → 64D)
    ↓
Linear Layer (64 → 256) + Swish
    ↓
Residual Block 1 (256 → 256)
    ↓
Residual Block 2 (256 → 256)
    ↓
... (10 residual blocks total)
    ↓
Residual Block 10 (256 → 256)
    ↓
Linear Layer (256 → 1)
    ↓
Output (scalar field)
```

### What the Networks Learn

**From Input → Output Mapping**:
- Given any point in 3D space (x, y, z) and time (cardiac phase)
- Networks learn to predict velocity (u, v, w) and pressure (p)
- This creates a **continuous representation** of the entire flow field

**Learning is Constrained By**:
1. **Wall Shear Stress Data** (CRITICAL - 50× weight)
   - CFD-computed WSS at 7,085 wall points
   - Forces networks to match velocity gradients at walls

2. **Physics Laws** (1× weight)
   - Navier-Stokes equations at 1,000 interior points
   - Ensures fluid dynamics conservation laws are satisfied

3. **Boundary Conditions** (20× weight)
   - No-slip condition: velocity = 0 at walls
   - Enforces physical constraints

4. **Pressure Data** (10× weight)
   - Matches predicted pressure with CFD pressure at walls

### Training Process

**Inputs During Training**:
- Wall coordinates: (x, y, z) at 7,085 points
- Interior coordinates: (x, y, z) at 1,000 points
- Cardiac phase: t ∈ {0.0 (diastolic), 1.0 (systolic)}
- Ground truth WSS: (WSS_x, WSS_y, WSS_z) from CFD
- Ground truth pressure: p from CFD

**What Networks Optimize**:
- Minimize weighted sum of 5 loss functions
- Learn smooth, physically-consistent flow field
- Match WSS patterns from CFD data

**Training Duration**: 15,000 epochs (~several hours on GPU)

---

## Training Methodology

### Non-Dimensional Formulation

**NEW in V4**: The codebase now uses proper non-dimensionalization with a Reynolds-number-based formulation, ensuring dimensional consistency and physical correctness.

#### Reference Scales (Auto-Computed)

The data loader automatically computes reference scales from the training data and fluid properties:

```python
# Physical Properties (standard blood)
rho = 1060 kg/m³     # Blood density
mu  = 0.0035 Pa·s    # Dynamic viscosity (Carreau-Yasuda μ_∞)

# Geometric Scale
L_ref = 0.05 m       # Characteristic length (vessel diameter)

# Derived from Data
P_char = max(|p|) across all training data  # ≈ 407 Pa for AS5
U_ref = sqrt(P_char / rho)                  # ≈ 0.62 m/s (characteristic velocity)
P_ref = rho × U_ref²                        # ≈ 407 Pa (by construction)
tau_ref = mu × U_ref / L_ref                # ≈ 0.0434 Pa (viscous stress scale)

# Reynolds Number
Re = rho × U_ref × L_ref / mu               # ≈ 9,383 (physiological range)
```

**Key Insight**: U_ref is derived from pressure data (not arbitrary), giving a physically-motivated characteristic velocity consistent with Bernoulli's principle.

#### Non-Dimensional Variables

All physical quantities are normalized:

```
Coordinates:  x_bar = (x - x_mean) / L_ref
Velocity:     u_bar = u / U_ref
Pressure:     p_bar = p / P_ref = p / (rho × U_ref²)
WSS (viscous):tau_bar = tau / tau_ref = tau / (mu × U_ref / L_ref)
Time:         t_bar = t × U_ref / L_ref  (for unsteady, if needed)
```

#### Non-Dimensional Equations

**Momentum (Navier-Stokes)**:
```
u_bar · ∇u_bar = -∇p_bar + (1/Re) × ∇²u_bar
```

**Continuity**:
```
∇ · u_bar = 0
```

**WSS (viscous scaling)**:
```
tau_bar_ij = ∂u_bar_i/∂x_bar_j + ∂u_bar_j/∂x_bar_i

WSS_bar = tangential(tau_bar · n)
```

**Key Property**: In the viscous-scaled WSS formulation, the explicit μ factor **cancels out** between the prediction (which uses ∂u_bar/∂x_bar) and the target (which is tau/tau_ref = tau/(mu×U/L)). This makes the WSS loss cleaner and more numerically stable.

#### Benefits of This Formulation

1. **Dimensional Consistency**: All terms in momentum equation are O(1), preventing one term from dominating
2. **Physical Correctness**: Uses actual blood properties (not unit values)
3. **Proper Re Formulation**: Physics loss is controlled by a single physical parameter (Re)
4. **Numerical Stability**: All fields normalized to similar magnitudes
5. **Research-Ready**: Standard formulation used in published CFD/PINN papers

#### Comparison with Previous Version

| Aspect | V3 (Old) | V4 (Current) |
|--------|----------|--------------|
| ρ | 1.0 kg/m³ (arbitrary) | 1060 kg/m³ (blood) |
| μ | 0.00125 Pa·s (arbitrary) | 0.0035 Pa·s (blood) |
| Pressure norm | Fixed 100 Pa | Auto: ~407 Pa (from data) |
| WSS norm | Fixed 1 Pa | Auto: ~0.043 Pa (viscous scale) |
| Physics form | Manual scaling factors | Clean Re-based formulation |
| Re | Not computed | 9,383 (physiological) |

#### Startup Log Output

When training starts, the reference scales are computed and displayed:

```
======================================================================
TAA-PINN TRAINING
======================================================================
Experiment: AS5_adaptive_physics
Description: Non-dimensional formulation: Re-based physics, auto-computed reference scales, blood properties
Geometry: AS5

----------------------------------------------------------------------
LOADING DATA (non-dimensional)
----------------------------------------------------------------------

  Non-dimensional reference scales:
    L_ref     = 0.0500 m
    U_ref     = 0.6196 m/s  (from max|p|=406.96 Pa)
    P_ref     = 406.96 Pa  (= rho * U_ref^2)
    tau_ref   = 0.043373 Pa  (= mu * U_ref / L_ref)
    Re        = 9382.8
    rho       = 1060.0 kg/m^3
    mu        = 0.003500 Pa.s
```

These values are stored in checkpoints for reproducibility.

---

## Training Methodology (Continued)

### Adaptive Physics Training Strategy

The current training approach implements several key improvements over standard PINN training to ensure stable convergence and prevent physics loss divergence:

**1. Physics Loss Annealing** (`physics_ramp_epochs: 2000`)
- **Problem**: Starting with full physics weight causes instability when networks haven't learned basic flow patterns
- **Solution**: Gradually ramp `λ_physics` from 0 to 1.0 over first 2000 epochs
- **Benefit**: Networks first learn to match wall data (WSS, pressure, BC), then progressively enforce interior physics
- **Formula**: `λ_effective = λ_physics × min(1.0, epoch / 2000)`

**2. Periodic Loss Renormalization** (`renorm_interval: 500`)
- **Problem**: Loss magnitudes change during training; initial normalization becomes stale
- **Solution**: Re-compute normalization factors every 500 epochs based on current raw loss values
- **Benefit**: Keeps all loss components contributing equally throughout training
- **Implementation**: `L_i / norm_i` where `norm_i` is updated periodically

**3. Dynamic Collocation Resampling** (`resample_collocation_interval: 1000`)
- **Problem**: Fixed interior points means physics is only enforced at same 2000 locations forever
- **Solution**: Re-sample interior collocation points every 1000 epochs
- **Benefit**: Better coverage of interior domain, prevents overfitting to specific point locations
- **Method**: Random offsets from wall points along normals (range: 0.05 to 0.5 in normalized coords)

**4. Validation Split** (`validation_split: 0.2`)
- **Problem**: No way to detect overfitting; all metrics computed on training data
- **Solution**: Hold out 20% of wall points for validation
- **Benefit**: Track generalization, use validation loss for early stopping
- **Implementation**: Fixed seed split at initialization, metrics computed every `validate_interval` epochs

**5. Consistent Coordinate Centering**
- **Problem**: Subsampling training data (10×) changes the coordinate mean, causing mismatch at evaluation
- **Solution**: Always center using full-dataset mean, even when subsampling
- **Benefit**: PINN sees same coordinate ranges during training and inference
- **Details**: Data loader computes full mean before subsampling, applies to subsampled data

**6. Immediate Best Model Saving**
- **Problem**: Previous versions only saved best model at checkpoint intervals (e.g., every 1000 epochs)
- **Solution**: Save `best_model.pt` immediately whenever validation loss improves
- **Benefit**: Don't lose the best model if training diverges between checkpoints

### Loss Weight Configuration

```yaml
# Fluid Properties (standard blood)
physics:
  rho: 1060.0                   # kg/m³ (blood density)
  mu: 0.0035                    # Pa·s (Carreau-Yasuda μ_inf)
  n_interior_points: 2000       # Collocation points
  resample_collocation_interval: 1000  # Resampling frequency

# Normalization (auto-computed from data + properties)
data:
  normalization:
    length_scale: 0.05          # L_ref (m) - vessel diameter
    # U_ref, P_ref, tau_ref, Re computed automatically

# Loss Weights
loss_weights:
  lambda_WSS: 50.0              # Wall shear stress (CRITICAL constraint)
  lambda_physics: 1.0           # Navier-Stokes (non-dimensional)
  lambda_BC_noslip: 20.0        # No-slip boundary condition
  lambda_pressure: 10.0         # Pressure matching
  physics_ramp_epochs: 2000     # Annealing duration
  
training:
  renorm_interval: 500          # Loss renormalization frequency
  validation_split: 0.2         # Fraction held out for validation
  epochs: 15000                 # Total training epochs
  save_interval: 2500           # Checkpoint frequency
```

**Note on Loss Magnitudes**: After non-dimensionalization, raw loss magnitudes are different from V3. The adaptive `loss_normalization: true` automatically adjusts normalization factors on the first epoch and every 500 epochs thereafter, so the same weights work well.

### Expected Training Behavior

**Healthy Training Signs**:
- Total loss decreases smoothly over first 5000 epochs
- Physics loss stays bounded (< 10.0 raw value by end of training)
- WSS loss < 0.1 by convergence
- Validation loss tracks training loss (no large gap)
- Correlation > 0.95 on validation set

**Warning Signs**:
- Physics loss diverging (> 1000) - indicates weights too low or annealing too fast
- Large gap between train/val loss - overfitting
- WSS loss > 1.0 after 10K epochs - geometry may need more capacity or different hyperparameters

---

## Dataset Description

## What Gets Visualized

### During Training (Real-time Monitoring)

**Console Output Every 100 Epochs**:
```
Epoch  5000 | Loss: 12345.678 | LR: 1.0e-04 | Time: 3600.0s
```

**Validation Every 500 Epochs**:
```
SYSTOLIC:
  Losses:
    Total:    12345.678
    WSS:      123.456
    Physics:  1234.567
    BC:       1.234
    Pressure: 12.345

  WSS Metrics:
    Relative L2: 0.0823     ← Target: < 0.10 (10% error)
    Correlation: 0.9234     ← Target: > 0.90
    MAE:         0.0345
    RMSE:        0.0512

  Physics Residuals:
    Momentum X: 1.234
    Momentum Y: 2.345
    Momentum Z: 3.456
    Continuity: 0.123
```

### After Training (Exported Files)

**1. VTK Files for ParaView Visualization** (recommended interval: every 1000 epochs)

Location: `experiments/AS5_baseline/vtk_outputs/`

**Files Generated**:
- `AS5_systolic_epoch_1000.vtu` - Systolic phase predictions
- `AS5_diastolic_epoch_1000.vtu` - Diastolic phase predictions
- `AS5_systolic_epoch_2000.vtu`
- ... (one pair per export interval)

**Visualizable Fields in ParaView**:

**Velocity Fields**:
- `velocity_u` - X-component of velocity (m/s)
- `velocity_v` - Y-component of velocity (m/s)
- `velocity_w` - Z-component of velocity (m/s)
- `velocity_magnitude` - Speed: √(u² + v² + w²)
- `velocity_vectors` - 3D velocity arrows

**Pressure Field**:
- `pressure` - Pressure distribution (Pa)
- `pressure_gradient` - ∇p for visualization

**Wall Shear Stress (WSS)**:
- `wss_predicted` - PINN-predicted WSS (Pa)
- `wss_true` - CFD ground truth WSS (Pa)
- `wss_error` - Absolute error: |WSS_pred - WSS_true|
- `wss_relative_error` - Relative error: |WSS_pred - WSS_true| / WSS_true

**Derived Quantities**:
- `vorticity` - Curl of velocity field (rotation)
- `strain_rate` - Rate of deformation tensor
- `helicity` - Measure of flow rotation

**Geometry**:
- `wall_normals` - Surface normal vectors
- `geometry_mask` - Labels different regions

### Visualization Examples

**1. Velocity Streamlines** (in ParaView):
```
Filters → Stream Tracer → Input: velocity_vectors
- Shows blood flow paths through aneurysm
- Can identify recirculation zones
```

**2. WSS Heatmap**:
```
Color by: wss_magnitude
- High WSS (red): > 4 Pa - potential damage zones
- Low WSS (blue): < 0.4 Pa - stagnation, thrombosis risk
```

**3. Pressure Contours**:
```
Contour: pressure levels
- Visualizes pressure drops
- Identifies stenotic regions
```

**4. Error Analysis**:
```
Color by: wss_error
- Shows where PINN predictions differ from CFD
- Validates model accuracy
```

### Key Visualizations for Publications

**Figure 1: Flow Field Comparison**
- Side-by-side: CFD velocity vs. PINN velocity
- Demonstrates learned flow patterns

**Figure 2: WSS Distribution**
- Heatmap of WSS on aneurysm surface
- Compare: CFD ground truth, PINN prediction, absolute error

**Figure 3: Training Curves**
- Loss vs. epoch for all loss components
- Shows convergence behavior

**Figure 4: Scatter Plot**
- WSS_predicted vs. WSS_true
- Diagonal line = perfect prediction
- Correlation coefficient shown

---

## Methodology

### Problem Statement

**Challenge**: Understanding blood flow dynamics in diseased aortas (TAAs) is critical for:
- Rupture risk assessment
- Treatment planning
- Patient-specific modeling

**Traditional Approaches**:
1. **CFD Simulation**: Accurate but computationally expensive, requires complete boundary conditions
2. **Direct ML**: Requires large training datasets (thousands of geometries)
3. **Standard PINN**: Needs sparse interior velocity measurements (hard to obtain clinically)

**Our Innovation**: Use dense wall shear stress (WSS) data to learn interior flow field

### Why This Works

**Key Insight**: Wall shear stress is the **velocity gradient at the wall**:

```
WSS = μ × (∂velocity/∂n)|_wall
```

Where:
- μ = fluid viscosity
- n = normal direction to wall

By matching WSS, we're effectively constraining the **derivative of velocity** at boundaries. Combined with physics equations (Navier-Stokes) in the interior, this uniquely determines the entire flow field!

### Methodology Steps

#### Step 1: Data Preparation

**Input Data** (from CFD simulations):
- Wall surface mesh: 70,843 points (subsampled to 7,085)
- Wall shear stress: WSS_x, WSS_y, WSS_z at each wall point
- Pressure: p at each wall point
- Two cardiac phases: systolic (peak flow) and diastolic (low flow)

**Preprocessing**:
1. Parse CSV files with CFD data
2. Normalize coordinates: center at origin, scale by characteristic length (5 cm)
3. Compute wall normal vectors using Open3D
4. Sample interior collocation points (offset from wall)

#### Step 2: Network Architecture

**Design Choices**:

1. **Fourier Feature Encoding**:
   - Maps input (x,y,z,t) to high-dimensional space (64D)
   - Enables learning of high-frequency WSS patterns
   - Proven to improve PINN performance (Tancik et al., 2020)

2. **Residual Blocks**:
   - 10 layers with skip connections
   - Improves gradient flow in deep networks
   - Prevents vanishing gradients

3. **Swish Activation**:
   - Smooth, non-monotonic: f(x) = x · sigmoid(x)
   - Better than ReLU for PINNs

4. **Separate Networks**:
   - One network per field (u, v, w, p)
   - Allows independent learning
   - Follows original PINN-wss architecture

#### Step 3: Loss Function Design (THE KEY INNOVATION)

**Total Loss**:
```python
L_total = λ_WSS × L_WSS             # 50.0
        + λ_physics × L_physics      # 1.0
        + λ_BC × L_BC                # 20.0
        + λ_pressure × L_pressure    # 10.0
```

**L_WSS (Wall Shear Stress Loss) - CRITICAL**:

```python
# At each wall point:
# 1. Predict velocity: u, v, w = Networks(x, y, z, t)
# 2. Compute velocity gradients via automatic differentiation:
#    ∂u/∂x, ∂u/∂y, ∂u/∂z, ∂v/∂x, ... (9 gradients total)
# 3. Compute stress tensor: τ_ij = μ(∂u_i/∂x_j + ∂u_j/∂x_i)
# 4. Compute traction: t = τ · n (n = wall normal)
# 5. Extract tangential component: WSS = t - (t·n)n
# 6. Compare with CFD: Loss = MSE(WSS_pred, WSS_true)
```

**Why This Works**:
- WSS depends on velocity **gradients**, not just values
- Gradients computed automatically by PyTorch
- Creates strong constraint on flow field near wall

**L_physics (Navier-Stokes Equations)**:

```python
# At interior points, enforce conservation laws:
# Momentum: ρ(u·∇u) = -∇p + μ∇²u
# Continuity: ∇·u = 0
# Residual = |equation| should be ≈ 0
```

**L_BC (Boundary Conditions)**:
```python
# No-slip at wall: u = v = w = 0
```

**L_pressure (Pressure Matching)**:
```python
# Match predicted pressure with CFD pressure at wall
```

#### Step 4: Training Strategy

**Optimizer**: Adam
- Learning rate: 1e-4
- β = (0.9, 0.99)
- Learning rate decay: 0.5 every 1000 epochs

**Batch Processing** (Memory-Efficient):
- Wall points: Process 1,000 at a time
- Interior points: Process 500 at a time
- Prevents GPU out-of-memory errors

**Training Loop** (10,000 epochs):
```
For each epoch:
    1. Zero gradients
    2. Compute all losses (batched)
    3. Backward pass (compute gradients)
    4. Gradient clipping (max_norm = 1.0)
    5. Optimizer step (update weights)
    6. Scheduler step (decay learning rate)
    7. Log progress
    8. Save checkpoints
```

**Validation** (every 500 epochs):
- Compute WSS metrics on full dataset
- Check physics residuals
- Save best model

#### Step 5: Validation Metrics

**Primary Metric: WSS Relative L2 Error**
```
Relative L2 = ||WSS_pred - WSS_true||₂ / ||WSS_true||₂
```

**Success Criteria**:
- **Minimum**: < 15% error
- **Good**: < 10% error, correlation > 0.9
- **Excellent**: < 5% error, correlation > 0.95

**Secondary Metrics**:
- Correlation coefficient between predicted and true WSS
- Mean Absolute Error (MAE)
- Root Mean Square Error (RMSE)
- Physics residuals (momentum, continuity)

### Comparison with Original PINN-wss

| Aspect | Original PINN-wss | Our TAA-PINN |
|--------|------------------|--------------|
| **Data Source** | VTK mesh files | CSV CFD exports |
| **Input Data** | 5 sparse interior velocity points | 7,085 dense wall WSS points |
| **Geometry** | Simple stenosis/aneurysm | Complex 3D TAA geometries |
| **Innovation** | Learn from sparse data | Learn from boundary gradients |
| **Phases** | Single snapshot | Two cardiac phases |
| **Architecture** | 9 layers, 200 neurons | 10 layers, 256 neurons, Fourier features |
| **Memory** | No batching | Batched for efficiency |

### Scientific Contributions

1. **Novel Constraint Method**: Using WSS (velocity gradients) instead of velocity values
2. **Clinical Relevance**: WSS is measurable, interior velocity is not
3. **Memory Efficiency**: Batched processing for large datasets
4. **Multi-Phase Learning**: Simultaneous systolic/diastolic modeling

---

## Dataset Description

### Source

TAA CFD data from patient-specific geometries simulating blood flow through thoracic aortic aneurysms.

### Geometries (6 Total)

| Code | Description | Diameter | β (Shape Parameter) | Type |
|------|-------------|----------|---------------------|------|
| **AS5** | 5cm Standard | 5.0 cm | 1.00 | Axisymmetric (Fusiform) |
| **PD5** | 5cm ASU | 5.0 cm | 1.78 | Posterior-Dominant (Saccular) |
| **AD5** | 5cm ASD | 5.0 cm | 0.56 | Anterior-Dominant (Saccular) |
| **AS6** | 6cm Standard | 6.0 cm | 1.00 | Axisymmetric (Fusiform) |
| **PD6** | 6cm ASU | 6.0 cm | 1.61 | Posterior-Dominant (Saccular) |
| **AD6** | 6cm ASD | 6.0 cm | 0.62 | Anterior-Dominant (Saccular) |

**Shape Parameter (β)**:
- β = 1.0: Symmetric aneurysm
- β > 1.0: Bulge on posterior (back) side
- β < 1.0: Bulge on anterior (front) side

### Cardiac Phases (2 per geometry)

1. **Systolic**: Peak cardiac contraction, maximum flow
2. **Diastolic**: Relaxation phase, minimum flow

**Total Dataset**: 12 files (6 geometries × 2 phases)

### Data Format

**CSV Structure**:
```
[Name]
a1

[Data]
X [ m ], Y [ m ], Z [ m ], Pressure [ Pa ], Velocity [ m s^-1 ],
Velocity u [ m s^-1 ], Velocity v [ m s^-1 ], Velocity w [ m s^-1 ],
Wall Shear [ Pa ], Wall Shear X [ Pa ], Wall Shear Y [ Pa ], Wall Shear Z [ Pa ]

0.0709, 0.0432, -0.0093, -22.86, 0.0, 0.0, 0.0, 0.0, 0.126, -0.082, -0.077, 0.055
...
```

**Fields Used**:
- **X, Y, Z**: 3D wall surface coordinates (meters)
- **Pressure**: Pressure at wall (Pascals)
- **Wall Shear X/Y/Z**: WSS vector components (Pascals)
- **Wall Shear**: WSS magnitude (Pascals)

**Fields Ignored** (all zeros at wall):
- Velocity, Velocity u/v/w: Zero due to no-slip condition

### Data Statistics (AS5 Diastolic, Subsampled)

```
Points: 7,085 wall points
Coordinate Range:
  X: [-2.30, 2.04] (normalized)
  Y: [-0.73, 0.56]
  Z: [-0.50, 0.50]

Pressure Range: [-1.03, 0.03] (normalized)
WSS Range: [0.01, 9.71] Pa (normalized)
WSS Mean: 0.94 Pa
WSS Std: 0.88 Pa
```

---

## Installation & Usage

### Prerequisites

**Environment**: Use existing `dl_env` conda environment
```bash
conda activate dl_env
```

**Required Packages** (already installed in dl_env):
- PyTorch 2.7.0 (CUDA 12.8)
- open3d 0.19.0
- vtk 9.5.2
- optuna 4.5.0
- pyyaml
- numpy, pandas

### Quick Start

**1. Navigate to Project**:
```bash
cd C:\Users\ajaoo\Documents\PINN-wss-workspace\PINN-wss\TAA-aneurysm
```

**2. Start Training**:
```bash
cd training
python train_single_geometry.py --config ../configs/AS5_config.yaml
```

**3. Monitor Progress**:
- Console logs every 100 epochs
- Validation every 500 epochs
- Checkpoints saved to `experiments/AS5_baseline/`

**4. Stop Training**:
- Press `Ctrl+C`
- Training will resume from last checkpoint

**5. Generate XY/XZ/YZ Plots for Papers**:
```bash
python utils/plot_2d_slices.py --data-dir "../data copy" --filename "5cm systolic.csv" --field wss_magnitude --output-dir "experiments/figures"
```

### Configuration

Edit `configs/AS5_config.yaml` to adjust:

**Training**:
- `epochs`: Number of training iterations (default: 10,000)
- `learning_rate`: Initial learning rate (default: 1e-4)
- `batch_size`: Not used (batching handled internally)

**Model**:
- `hidden_dim`: Network width (default: 256)
- `num_layers`: Network depth (default: 10)
- `num_frequencies`: Fourier features (default: 32)

**Loss Weights**:
- `lambda_WSS`: WSS loss weight (default: 50.0) ← CRITICAL
- `lambda_physics`: Physics loss (default: 1.0)
- `lambda_BC_noslip`: No-slip BC (default: 20.0)
- `lambda_pressure`: Pressure matching (default: 10.0)

**Data**:
- `subsample_factor`: Downsample wall points (default: 10)
- `n_interior_points`: Interior collocation points (default: 1,000)

### Output Files

**During Training**:
```
experiments/AS5_baseline/
├── config.yaml                   # Saved configuration
├── checkpoint_epoch_1000.pt      # Model checkpoint
├── checkpoint_epoch_2000.pt
├── ...
├── best_model.pt                 # Best validation loss
└── final_model.pt                # Final trained model
```

**Training Log** (console output):
```
Epoch   100 | Loss: 452123.456 | LR: 1.0e-04 | Time: 120.3s
Epoch   200 | Loss: 345678.901 | LR: 1.0e-04 | Time: 240.6s
...
Epoch   500 | Loss: 123456.789 | LR: 1.0e-04 | Time: 601.5s

VALIDATION - Epoch 500
SYSTOLIC:
  Losses:
    Total:    123456.789
    WSS:      234.567
    ...
  WSS Metrics:
    Relative L2: 0.1234
    Correlation: 0.8765
    ...
```

### Resume Training

```bash
# Training automatically resumes from last checkpoint
python train_single_geometry.py --config ../configs/AS5_config.yaml
```

To manually load a checkpoint, modify the training script to call:
```python
trainer.load_checkpoint('experiments/AS5_baseline/checkpoint_epoch_5000.pt')
```

---

## Project Structure

```
TAA-aneurysm/
│
├── src/                              # All source code (Python package)
│   ├── __init__.py
│   ├── data/                         # Data loading and preprocessing
│   │   ├── __init__.py
│   │   └── loader.py                # Parse CSV, normalize, create tensors
│   ├── models/                       # Neural network architectures
│   │   ├── __init__.py
│   │   ├── networks.py              # Main network classes (TAANet, Net2_u/v/w/p)
│   │   ├── fourier.py              # Fourier feature encoding
│   │   └── blocks.py               # Residual block with skip connections
│   ├── losses/                       # Physics-informed loss functions
│   │   ├── __init__.py
│   │   ├── wss.py                   # WSS matching (critical innovation)
│   │   ├── physics.py              # Navier-Stokes equations
│   │   └── boundary.py             # No-slip BC, pressure matching
│   ├── training/                     # Training loop
│   │   ├── __init__.py
│   │   └── trainer.py              # TAATrainer class and main entry point
│   └── utils/                        # Utility functions
│       ├── __init__.py
│       ├── geometry.py              # Wall normals, interior sampling
│       └── plotting.py             # CFD vs PINN comparison plots
│
├── configs/                          # YAML configuration files (one per geometry)
│   ├── AS5_config.yaml
│   ├── AD5_config.yaml
│   ├── PD5_config.yaml
│   ├── AS6_config.yaml
│   ├── AD6_config.yaml
│   └── PD6_config.yaml
│
├── data/                             # CFD data (12 CSV files, 6 geometries x 2 phases)
│
├── experiments/                      # Training outputs (one subdir per geometry)
│   └── AS5_adaptive_physics/
│       ├── config.yaml              # Saved config snapshot
│       ├── best_model.pt            # Best model during training
│       ├── final_model.pt           # Model at end of training
│       ├── loss_history.csv         # Epoch-level metrics
│       └── loss_curves.png          # Training curves plot
│
├── scripts/                          # Shell launch scripts
│   └── run_training.sh             # GPU training launcher
│
├── logs/                             # Training log files (gitignored)
├── requirements.txt                  # Python dependencies
├── .gitignore
└── README.md
```

### Key Files Explained

**1. `csv_loader.py`** (326 lines)
- Parses TAA CSV files with custom header format
- Normalizes coordinates and physical quantities
- Computes statistics
- Creates PyTorch tensors on GPU

**2. `src/utils/geometry.py`**
- Uses Open3D to compute wall normal vectors
- Samples interior collocation points
- Validates normal vector quality

**3. `src/models/networks.py`**
- Defines TAANet class with Fourier features + residual blocks
- Creates all 4 networks (u, v, w, p)
- Kaiming initialization for stable training

**4. `src/losses/wss.py`** - **MOST IMPORTANT**
- Computes 9 velocity gradients via automatic differentiation
- Builds non-dimensional stress tensor: `tau_bar = ∇u_bar + (∇u_bar)ᵀ`
- **Viscous scaling**: μ cancels out (both pred and target scaled by tau_ref = μ×U/L)
- Extracts tangential component (WSS)
- Compares with CFD ground truth

**5. `src/losses/physics.py`**
- Enforces 3D non-dimensional Navier-Stokes: `u·∇u + ∇p - (1/Re)×∇²u = 0`
- Enforces continuity (divergence-free): `∇·u = 0`
- **Single parameter**: Reynolds number Re ≈ 9,383
- Clean formulation with no manual scaling factors

**6. `src/training/trainer.py`**
- TAATrainer class: main training orchestration
- Auto-computes reference scales (U_ref, P_ref, tau_ref, Re) from data
- Batched loss computation for memory efficiency
- Validation and checkpointing (stores ref_scales for reproducibility)
- Progress logging

**7. `src/data/loader.py`**
- TAADataLoader: loads CFD CSV files
- **Two-pass loading**: First pass finds max(|p|) to compute U_ref
- Normalizes all data with physically-consistent scales
- Returns Re and all reference scales for training

---

## Results & Validation

### Expected Results After Training

**Convergence Behavior**:
- Epochs 0-1000: Rapid loss decrease (physics learning)
- Epochs 1000-5000: Steady improvement (WSS refinement)
- Epochs 5000-10000: Fine-tuning (plateau near optimal)

**Target Metrics** (after 10,000 epochs):
```
WSS Relative L2 Error: < 10%
WSS Correlation: > 0.90
Physics Residuals: < 1.0 (normalized)
Boundary Condition Violation: < 1e-3
```

### Validation Checklist

**1. Visual Inspection** (in ParaView):
- [ ] Velocity field is smooth (no oscillations)
- [ ] Flow direction matches expectations
- [ ] WSS heatmap shows realistic patterns
- [ ] High WSS at flow impingement zones
- [ ] Low WSS in recirculation zones
- [ ] Pressure decreases along flow direction

**2. Quantitative Metrics**:
- [ ] WSS error < 10%
- [ ] WSS correlation > 0.9
- [ ] Physics residuals converged
- [ ] No gradient explosions (check NaN in logs)

**3. Physical Plausibility**:
- [ ] Maximum velocity < 2 m/s (physiological range for aortic flow)
- [ ] Reynolds number ≈ 9,000-10,000 (matches typical aortic Re)
- [ ] Pressure gradient negative in flow direction
- [ ] Velocity magnitude decreases from systole to diastole
- [ ] Recirculation zones present in aneurysm bulge
- [ ] WSS values in range 0-5 Pa (typical for aortic wall)

### Common Issues & Solutions

**Issue 1: GPU Out of Memory**
```
Solution: Reduce batch sizes in config.yaml
  wall_batch_size: 1000 → 500
  n_interior_points: 1000 → 500
```

**Issue 2: Loss Not Decreasing**
```
Solution: Adjust loss weights
  Increase lambda_WSS if WSS error high
  Increase lambda_physics if residuals high
```

**Issue 3: Training Unstable (NaN)**
```
Solution:
  - Reduce learning_rate: 1e-4 → 5e-5
  - Enable gradient_clip: 1.0
  - Check data normalization
```

**Issue 4: WSS Correlation Low**
```
Solution:
  - Increase num_frequencies for Fourier encoding
  - Increase network depth (num_layers)
  - Train longer (more epochs)
```

---

## Advanced Usage

### Hyperparameter Tuning with Optuna

```python
# TODO: Implement in hyperparameter_tuning.py
import optuna

def objective(trial):
    lambda_wss = trial.suggest_float("lambda_wss", 10.0, 100.0)
    learning_rate = trial.suggest_float("lr", 1e-5, 1e-3, log=True)

    # Train model with these params
    trainer = TAATrainer(config)
    trainer.train(epochs=2000)  # Shorter for tuning

    # Return validation metric
    return trainer.best_loss

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=100)
```

### Multi-Geometry Training

```python
# TODO: Extend train_single_geometry.py
# Load all 6 geometries simultaneously
# Sample batches from different geometries
# Learn unified representation
```

### Uncertainty Quantification

```python
# TODO: Implement Bayesian PINN
# Add dropout layers for Monte Carlo sampling
# Compute prediction uncertainty
# Visualize confidence intervals
```

---

## References

### Original PINN-wss Paper
```
Arzani, A., Wang, J.X., & D'Souza, R.M. (2021).
"Uncovering near-wall blood flow from sparse data with physics-informed neural networks"
arXiv:2104.08249
```

### Fourier Features for PINNs
```
Tancik, M., et al. (2020).
"Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains"
NeurIPS 2020
```

### Physics-Informed Neural Networks
```
Raissi, M., Perdikaris, P., & Karniadakis, G.E. (2019).
"Physics-informed neural networks: A deep learning framework for solving forward and inverse problems"
Journal of Computational Physics, 378, 686-707
```

### TAA Hemodynamics
```
[Add relevant TAA/CFD papers from your dataset source]
```

---

## Citation

If you use this code, please cite:

```bibtex
@software{taa_pinn_2026,
  title={TAA-PINN: Physics-Informed Neural Networks for Thoracic Aortic Aneurysm Analysis},
  author={[Your Name]},
  year={2026},
  url={https://github.com/[your-repo]/TAA-PINN}
}
```

---

## License

[Specify license - e.g., MIT, Apache 2.0]

---

## Contact

For questions or issues:
- Open an issue on GitHub
- Email: [your email]

---

## Acknowledgments

- Original PINN-wss implementation by Amirhossein Arzani
- TAA CFD dataset from [specify source]
- Claude Code assistance in implementation

---

## Multi-GPU Training on HPC (brosnan.coventry.ac.uk)

### Current Setup

**Environment**: `deep_tf` conda environment (has open3d installed)

### Running Training Jobs

**GPU0: Resume AS5 from epoch 1000**
```bash
nohup bash -lc "source ~/miniconda3/etc/profile.d/conda.sh; conda activate deep_tf; CUDA_VISIBLE_DEVICES=0 python training/train_single_geometry.py --config configs/AS5_config.yaml --resume experiments/AS5_baseline/checkpoint_epoch_1000.pt" > hpc/logs/as5_resume_gpu0.log 2>&1 &
```

**GPU1: Run all remaining geometries (AD5, PD5, AS6, AD6, PD6)**
```bash
CUDA_VISIBLE_DEVICES=1 GEOMS_LIST="AD5 PD5 AS6 AD6 PD6" nohup bash hpc/run_all_geometries_detached.sh > hpc/logs/others_gpu1.log 2>&1 &
```

### Monitoring Running Jobs

**Check log files:**
```bash
# AS5 on GPU0
tail -f hpc/logs/as5_resume_gpu0.log

# Other geometries on GPU1
tail -f hpc/logs/others_gpu1.log

# Check last 50 lines
tail -50 hpc/logs/as5_resume_gpu0.log
```

**Check GPU usage:**
```bash
nvidia-smi

# Watch GPU usage in real-time (updates every 2 seconds)
watch -n 2 nvidia-smi
```

**Check running processes:**
```bash
ps aux | grep train_single_geometry

# Check background jobs
jobs -l
```

**Kill a job if needed:**
```bash
# Find the process ID (PID)
ps aux | grep train_single_geometry

# Kill it
kill <PID>

# Force kill if not responding
kill -9 <PID>
```

### Saved Metrics and Outputs

**During Training (every epoch):**
- `experiments/<geometry>_baseline/loss_history.csv` - Full training metrics
  - Columns: epoch, total, wss, physics, bc_noslip, pressure, residual_momentum_x/y/z, residual_continuity, lr

**Checkpoints (every 1000 epochs):**
- `experiments/<geometry>_baseline/checkpoint_epoch_1000.pt`
- `experiments/<geometry>_baseline/checkpoint_epoch_2000.pt`
- Contains: network weights, optimizer state, scheduler state, loss history, epoch number

**Final Outputs:**
- `experiments/<geometry>_baseline/loss_history.csv` - Complete training log
- `experiments/<geometry>_baseline/loss_curves.png` - Training curve plots
- `experiments/<geometry>_baseline/best_model.pt` - Best model by validation loss
- `experiments/<geometry>_baseline/final_model.pt` - Final trained model

**Accessing Metrics:**
```python
import pandas as pd

# Load training history
df = pd.read_csv('experiments/AS5_baseline/loss_history.csv')

# Plot WSS loss over time
import matplotlib.pyplot as plt
plt.plot(df['epoch'], df['wss'])
plt.xlabel('Epoch')
plt.ylabel('WSS Loss')
plt.yscale('log')
plt.show()

# Find best epoch
best_epoch = df.loc[df['total'].idxmin(), 'epoch']
print(f"Best epoch: {best_epoch}")
```

**Resume from Specific Checkpoint:**
```bash
python training/train_single_geometry.py \
  --config configs/AS5_config.yaml \
  --resume experiments/AS5_baseline/checkpoint_epoch_5000.pt
```

---

**Last Updated**: February 2026

**Status**: ✅ Fully Functional - Ready for Training
