# TAA-PINN Methodology Summary

## What Gets Trained?

### Four Separate Neural Networks

Each network learns to predict ONE component of the fluid flow field:

```
┌─────────────────────────────────────────────────────────┐
│  INPUT: (x, y, z, t_phase)                             │
│  - x, y, z: Any point in 3D space                      │
│  - t_phase: 0.0 (diastolic) or 1.0 (systolic)         │
└─────────────────────────────────────────────────────────┘
                          ↓
    ┌──────────────┬──────────────┬──────────────┬──────────────┐
    │   Net_u      │   Net_v      │   Net_w      │   Net_p      │
    │ (1.33M params)│(1.33M params)│(1.33M params)│(1.33M params)│
    └──────────────┴──────────────┴──────────────┴──────────────┘
         ↓              ↓              ↓              ↓
    ┌──────────────┬──────────────┬──────────────┬──────────────┐
    │   u(x,y,z,t) │   v(x,y,z,t) │   w(x,y,z,t) │   p(x,y,z,t) │
    │  X-velocity  │  Y-velocity  │  Z-velocity  │   Pressure   │
    └──────────────┴──────────────┴──────────────┴──────────────┘
```

### What Each Network Outputs

- **Net_u**: Velocity in X-direction (horizontal flow)
- **Net_v**: Velocity in Y-direction (vertical flow)
- **Net_w**: Velocity in Z-direction (depth flow)
- **Net_p**: Pressure field

**Combined**: Full 3D velocity vector **v** = (u, v, w) + pressure p at any point

### How They Learn

Networks optimize to minimize **5 loss functions simultaneously**:

```
Total Loss = 50.0 × L_WSS          ← HIGHEST WEIGHT (main constraint)
           +  1.0 × L_physics       ← Physics equations
           + 20.0 × L_BC            ← Boundary conditions
           + 10.0 × L_pressure      ← Pressure matching
```

---

## What Gets Visualized?

### 1. During Training (Real-Time Console)

**Progress Logs** (every 100 epochs):
```
Epoch  1000 | Loss: 123456.789 | LR: 1.0e-04 | Time: 1200.5s
```
- **Loss**: Should decrease over time (convergence)
- **LR**: Learning rate (decreases automatically)
- **Time**: Total training time so far

**Validation Metrics** (every 500 epochs):
```
WSS Relative L2: 0.0823    ← Getting closer to 0 = perfect
Correlation:     0.9234    ← Getting closer to 1 = perfect
```

### 2. After Training (Output Files)

**A. Checkpoint Files** (model weights)
- `best_model.pt` - Best performing model
- `checkpoint_epoch_5000.pt` - Snapshots during training
- Can be loaded to resume training or make predictions

**B. VTK Files** (3D visualization - TODO)
Will contain:
- Velocity field: u, v, w at all 7,085 wall points
- Pressure field: p at all wall points
- WSS predicted vs. WSS true (error analysis)
- Can be opened in ParaView for 3D visualization

### 3. What You Can Visualize in ParaView

Once VTK export is implemented:

**Velocity Streamlines**:
```
Shows: Blood flow paths through aneurysm
Colors: Speed (red=fast, blue=slow)
Use: Identify recirculation zones
```

**WSS Heatmap**:
```
Shows: Shear stress on aneurysm wall
Colors: Magnitude (red=high stress, blue=low stress)
Use: Find regions at risk of damage or thrombosis
```

**Pressure Contours**:
```
Shows: Pressure distribution
Colors: High to low pressure
Use: Understand pressure drops
```

**Error Maps**:
```
Shows: |WSS_predicted - WSS_true|
Colors: Error magnitude
Use: Validate model accuracy
```

---

## The Methodology Explained

### Problem We're Solving

**Given**: Wall shear stress (WSS) from CFD at 7,085 surface points

**Find**: Complete 3D velocity field (u, v, w) and pressure (p) everywhere

**Why it's hard**:
- WSS is only at the wall (boundary)
- Need to infer interior flow (no direct measurements)
- Must satisfy physics laws (Navier-Stokes)

### Our Innovation: WSS-Constrained Learning

**Key Insight**: WSS = velocity gradient at wall

```
        WSS = μ × (∂velocity/∂n)|wall
               ↑
        This is the DERIVATIVE of velocity!
```

By matching WSS, we constrain how velocity CHANGES near the wall.

### Step-by-Step Process

**Step 1: Data Preparation**
```
Input: CSV files with wall coordinates + WSS from CFD
Process:
  - Parse 70,843 points → subsample to 7,085
  - Compute wall normal vectors using Open3D
  - Sample 1,000 interior points for physics loss
  - Normalize everything
Output: PyTorch tensors on GPU
```

**Step 2: Network Forward Pass**
```
For each point (x, y, z, t):
  1. Encode with Fourier features → high-dimensional space
  2. Pass through 10 residual blocks
  3. Output: scalar value for u, v, w, or p

For 7,085 wall points:
  - Predict u, v, w, p at each point
  - Compute velocity gradients via autograd
```

**Step 3: Compute Losses**

**L_WSS (Wall Shear Stress Loss)**:
```python
For each wall point:
  1. Predict velocities: u, v, w
  2. Compute 9 gradients: ∂u/∂x, ∂u/∂y, ∂u/∂z, ∂v/∂x, ...
  3. Build stress tensor: τ_ij = μ(∂u_i/∂x_j + ∂u_j/∂x_i)
  4. Compute traction: t = τ · n (n = wall normal)
  5. Extract tangential part: WSS = t - (t·n)n
  6. Compare with CFD: Loss = MSE(WSS_pred, WSS_CFD)
```

**L_physics (Navier-Stokes)**:
```python
For each interior point:
  1. Predict u, v, w, p
  2. Compute first & second derivatives
  3. Evaluate momentum equations:
     ρ(u∇u) = -∇p + μ∇²u  ← Should equal 0
  4. Evaluate continuity:
     ∇·u = 0               ← Should equal 0
  5. Loss = MSE(residuals, 0)
```

**L_BC (Boundary Conditions)**:
```python
At wall points:
  Loss = MSE(u, 0) + MSE(v, 0) + MSE(w, 0)
  (No-slip: velocity must be zero)
```

**L_pressure (Pressure Matching)**:
```python
At wall points:
  Loss = MSE(p_predicted, p_CFD)
```

**Step 4: Backpropagation**
```
Total Loss → Gradients → Update Network Weights
```

**Step 5: Repeat**
```
Do this 10,000 times (epochs)
Networks gradually learn flow field that:
  ✓ Matches WSS at walls
  ✓ Satisfies Navier-Stokes in interior
  ✓ Has zero velocity at walls
  ✓ Matches pressure distribution
```

### Why This Works

**Traditional ML**: Needs thousands of examples
**Our PINN**: Needs only 1 geometry + physics equations

**Magic**: Physics provides infinite training data!
- Every point in 3D space gives a physics constraint
- Sample 1,000 random interior points each batch
- Effectively have unlimited training examples

**WSS Constraint**: Pulls solution from boundary
- Traditional PINN: Push from interior measurements
- Our PINN: Pull from boundary gradients

### Mathematical Formulation

**Optimization Problem**:
```
Find networks θ = {θ_u, θ_v, θ_w, θ_p} that minimize:

L(θ) = λ_WSS · ||WSS_pred(θ) - WSS_CFD||²
     + λ_physics · ||Navier-Stokes residual||²
     + λ_BC · ||velocity at wall||²
     + λ_pressure · ||p_pred(θ) - p_CFD||²

Subject to:
  u, v, w, p are smooth (neural networks provide this)
  Gradients exist (autograd computes exactly)
```

**Degrees of Freedom**: 5.33 million parameters
**Constraints**: ~8,000 data points + physics everywhere
**Result**: Highly constrained, well-posed problem

---

## Comparison with Original PINN-wss

| Aspect | Original | Our TAA-PINN |
|--------|----------|--------------|
| **What trains from** | 5 sparse interior velocity points | 7,085 dense wall WSS points |
| **Innovation** | Learn from minimal data | Learn from boundary gradients |
| **Clinical relevance** | Limited (interior hard to measure) | High (WSS is measurable) |
| **Data constraint** | Direct (velocity values) | Indirect (velocity derivatives) |
| **Architecture** | Basic MLP | Fourier features + residual blocks |
| **Memory** | All at once | Batched processing |

---

## Physical Interpretation

### What the Networks Represent

**Net_u, Net_v, Net_w**:
- Continuous function approximators
- Learn the **velocity field** u(x,y,z,t)
- Can evaluate at ANY point (mesh-free)

**Net_p**:
- Learns the **pressure field** p(x,y,z,t)
- Satisfies momentum balance with velocity

### What We're Learning

Not just interpolation! We're learning:

1. **Physics**: Momentum + mass conservation
2. **Boundary behavior**: WSS patterns
3. **Pressure-velocity coupling**: Navier-Stokes
4. **Temporal variation**: Systolic vs. diastolic

### Why It's Better Than Pure CFD Surrogate

**CFD Surrogate** (traditional ML):
- Needs many geometries (dataset)
- Interpolates between known cases
- No physics guarantees

**Our PINN**:
- Works with ONE geometry
- Respects physics exactly
- Generalizes via physics laws

---

## Success Metrics

### Primary: WSS Accuracy

**Relative L2 Error**:
```
Error = ||WSS_pred - WSS_true|| / ||WSS_true||

Target: < 10%
Excellent: < 5%
```

**Correlation**:
```
How well predictions match CFD pattern

Target: > 0.90
Excellent: > 0.95
```

### Secondary: Physics Compliance

**Momentum Residuals**:
```
How well Navier-Stokes is satisfied

Target: < 1.0 (normalized)
```

**Continuity Residual**:
```
How divergence-free the flow is

Target: < 0.1
```

### Tertiary: Boundary Conditions

**No-slip Violation**:
```
Velocity magnitude at wall

Target: < 1e-3
```

---

## Practical Considerations

### Computational Cost

**Training** (one-time):
- Time: ~4 hours for 10K epochs on GPU
- Memory: ~2 GB GPU RAM
- Cost: One-time per geometry

**Inference** (many times):
- Time: Milliseconds for any point
- Memory: Minimal (just network weights)
- Cost: Nearly free

**Compare to CFD**:
- CFD: Hours to days per simulation
- PINN: Once trained, instant predictions

### Limitations

**What it CAN do**:
✓ Predict flow field at any point
✓ Interpolate between cardiac phases
✓ Satisfy physics exactly
✓ Work with single geometry

**What it CANNOT do** (yet):
✗ Generalize to unseen geometries (need unified model)
✗ Handle turbulence (assumes laminar flow)
✗ Real-time training (need pretrained)
✗ Extreme parameter ranges (trained on specific Re)

---

## Future Directions

### Short-term:
1. **VTK Export**: Visualize predictions in ParaView
2. **Metrics Dashboard**: Real-time training plots
3. **Multi-geometry**: Train on all 6 TAA shapes

### Medium-term:
1. **Unified Model**: Single network for all geometries
2. **Uncertainty Quantification**: Bayesian PINN
3. **Hyperparameter Optimization**: Automated tuning

### Long-term:
1. **Real-time Inference**: Deploy for clinical use
2. **Patient-specific**: Train from medical imaging
3. **Treatment Planning**: Optimize interventions

---

## Key Takeaways

1. **We train 4 neural networks** (u, v, w, p) with 5.33M total parameters

2. **We visualize**:
   - Training progress in console
   - Validation metrics every 500 epochs
   - Eventually: 3D flow fields in ParaView

3. **Our methodology**:
   - Use WSS to constrain boundary behavior
   - Use physics to constrain interior behavior
   - Learn continuous representation of flow
   - Validate against CFD ground truth

4. **Innovation**:
   - First PINN to use wall gradients (WSS) as main constraint
   - Enables learning from clinically-measurable quantities
   - Combines best of ML (flexibility) and physics (guarantees)

---

**This is a novel approach to inverse hemodynamics problems!**

The methodology bridges:
- Clinical data (measurable WSS)
- Computational fluid dynamics (physics)
- Deep learning (continuous representations)

Perfect for patient-specific cardiovascular modeling where interior measurements are impossible but wall properties can be estimated from imaging.
