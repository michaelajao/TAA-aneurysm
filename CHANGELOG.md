# TAA-PINN Changelog

## Version 4.0 - Full Non-Dimensionalization (Feb 16, 2026)

### Major Changes

**Non-dimensional formulation with Reynolds number-based physics**

This version implements proper non-dimensionalization for research-grade numerical consistency and physical correctness.

### Key Improvements

1. **Correct Blood Properties**
   - Density: ρ = 1060 kg/m³ (was 1.0)
   - Viscosity: μ = 0.0035 Pa·s (was 0.00125)
   - Based on Carreau-Yasuda model (μ_∞ for high shear rates)

2. **Auto-Computed Reference Scales**
   - U_ref derived from data: sqrt(max|p| / rho) ≈ 0.62 m/s
   - P_ref = ρ × U_ref² ≈ 407 Pa
   - tau_ref = μ × U_ref / L_ref ≈ 0.0434 Pa
   - **Reynolds number**: Re ≈ 9,383 (physiological range)

3. **Clean Physics Formulation**
   - Momentum: `u·∇u + ∇p - (1/Re)×∇²u = 0`
   - Single parameter (Re) instead of 6 scaling factors
   - Dimensionally consistent throughout

4. **Viscous-Scaled WSS**
   - Both prediction and target scaled by tau_ref
   - Explicit μ factor cancels out
   - More numerically stable

### Files Modified

- `configs/AS5_config.yaml` - Updated fluid properties
- `src/data/loader.py` - Two-pass loading, auto-compute scales
- `src/losses/physics.py` - Re-based Navier-Stokes
- `src/losses/wss.py` - Viscous scaling (μ cancels)
- `src/losses/boundary.py` - Non-dim velocity constraint
- `src/training/trainer.py` - Store ref_scales in checkpoints
- `src/utils/plotting.py` - Use ref_scales for de-normalization
- `README.md` - Comprehensive documentation update

### Validation

Tested with 100-epoch run:
- ✅ Re computed correctly: 9382.8
- ✅ All loss components finite and decreasing
- ✅ Loss normalization auto-adjusted
- ✅ Checkpoints saved with ref_scales
- ✅ No dimensional inconsistencies

### Migration Notes

**Breaking Changes:**
- Old checkpoints (V3) are incompatible with V4 code
- Config files need `rho` and `mu` updated
- Must remove `pressure_scale`, `wss_scale`, `velocity_scale` from configs

**Backward Compatibility:**
- None - this is a major refactor
- Re-train all geometries from scratch

### Performance

Training speed unchanged: ~1.4 sec/epoch on GPU
Expected convergence behavior similar to V3

---

## Version 3.0 - Adaptive Physics Training (Feb 2026)

### Key Features
- Physics loss annealing (2000 epochs ramp)
- Periodic loss renormalization (every 500 epochs)
- Dynamic collocation resampling (every 1000 epochs)
- Validation split (20%)
- Immediate best model saving

### Fixed Issues
- Physics loss divergence (λ_physics restored to 1.0)
- Coordinate centering inconsistency
- Loss normalization staleness

---

## Version 2.0 - Initial Release

Basic PINN implementation with WSS constraints
