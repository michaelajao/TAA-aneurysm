# TAA-PINN Quick Start Guide

## 🚀 Ready to Train in 3 Steps

### Step 1: Activate Environment
```bash
conda activate dl_env
```

### Step 2: Navigate to Training Directory
```bash
cd C:\Users\ajaoo\Documents\PINN-wss-workspace\PINN-wss\TAA-aneurysm\training
```

### Step 3: Start Training
```bash
python train_single_geometry.py --config ../configs/AS5_config.yaml
```

That's it! Training will begin immediately.

---

## 📄 Generate 2D XY/XZ/YZ Paper Plots

From the `TAA-aneurysm` folder, run:

```bash
python utils/plot_2d_slices.py --data-dir "../data copy" --filename "5cm systolic.csv" --field wss_magnitude --output-dir "experiments/figures"
```

This exports both PNG and SVG versions with three panels:
- XY plane
- XZ plane
- YZ plane

You can change `--field` to `pressure`, `wss_x`, `wss_y`, or `wss_z`.

---

## 📊 What You'll See

### During Training (Console Output)

```
======================================================================
TAA-PINN TRAINING
======================================================================

Experiment: AS5_baseline
Description: Baseline training for AS5 geometry with standard hyperparameters
Geometry: AS5
Device: cuda

----------------------------------------------------------------------
LOADING DATA
----------------------------------------------------------------------
Loading systolic: 5cm systolic.csv
  Wall points: 7085
  Interior points: 1000

Loading diastolic: 5cm diastolic.csv
  Wall points: 7085
  Interior points: 1000

----------------------------------------------------------------------
INITIALIZING NETWORKS
----------------------------------------------------------------------
  Net_u: 1,332,737 parameters
  Net_v: 1,332,737 parameters
  Net_w: 1,332,737 parameters
  Net_p: 1,332,737 parameters
  Total: 5,330,948 parameters

======================================================================
STARTING TRAINING
======================================================================

Epoch   100 | Loss: 452123.456 | LR: 1.0e-04 | Time: 120.3s
Epoch   200 | Loss: 345678.901 | LR: 1.0e-04 | Time: 240.6s
Epoch   300 | Loss: 234567.890 | LR: 1.0e-04 | Time: 360.9s
...
```

### Every 500 Epochs (Validation)

```
======================================================================
VALIDATION - Epoch 500
======================================================================

SYSTOLIC:
  Losses:
    Total:    123456.789
    WSS:      234.567      ← Should decrease
    Physics:  12345.678
    BC:       0.123
    Pressure: 12.345

  WSS Metrics:
    Relative L2: 0.1234    ← Target: < 0.10 (10%)
    Correlation: 0.8765    ← Target: > 0.90
    MAE:         0.0345
    RMSE:        0.0512

  Physics Residuals:
    Momentum X: 1.234      ← Should decrease
    Momentum Y: 2.345
    Momentum Z: 3.456
    Continuity: 0.123
```

---

## ⏱️ Training Time

**Expected Duration**:
- **10,000 epochs**: ~3-5 hours on GPU
- **Checkpoint saved**: Every 1000 epochs
- **Can resume**: Training resumes from last checkpoint if interrupted

**Stop Training Anytime**: Press `Ctrl+C`

---

## 📁 Output Files

All results saved to:
```
experiments/AS5_baseline/
├── config.yaml                  # Your training configuration
├── checkpoint_epoch_1000.pt     # Model weights at epoch 1000
├── checkpoint_epoch_2000.pt     # Model weights at epoch 2000
├── ...
├── best_model.pt                # Best validation performance
└── final_model.pt               # Final trained model
```

---

## 🎯 Success Indicators

### Good Training:
✅ Loss decreases steadily
✅ WSS Relative L2 < 10% after 10K epochs
✅ WSS Correlation > 0.90
✅ No NaN values in logs

### Issues to Watch:
❌ Loss increases → Reduce learning rate
❌ Loss = NaN → Check data normalization
❌ GPU out of memory → Reduce batch sizes in config

---

## 🔧 Common Adjustments

### Train Longer
```yaml
# configs/AS5_config.yaml
training:
  epochs: 20000  # Change from 10000
```

### Reduce Memory Usage
```yaml
training:
  wall_batch_size: 500      # Reduce from 1000
physics:
  n_interior_points: 500    # Reduce from 1000
```

### Focus More on WSS
```yaml
loss_weights:
  lambda_WSS: 100.0         # Increase from 50.0
```

---

## 📈 After Training

### View Results in ParaView (TODO - Future Feature)
```bash
# Export to VTK (feature to be implemented)
python utils/export_to_vtk.py --checkpoint experiments/AS5_baseline/best_model.pt
```

### Analyze Performance
```bash
# Compute detailed metrics (feature to be implemented)
python utils/analyze_results.py --checkpoint experiments/AS5_baseline/best_model.pt
```

---

## 🆘 Need Help?

1. **Check logs**: Look for error messages in console output
2. **Read README.md**: Comprehensive documentation
3. **Verify environment**: `conda list | grep torch` should show PyTorch 2.7.0
4. **Test data**: Data files should be in `../data copy/` directory

---

## 📚 Next Steps

After successful training on AS5:

1. **Train other geometries**:
   - Copy `AS5_config.yaml` → `PD5_config.yaml`
   - Update geometry and file names
   - Train again

2. **Compare geometries**:
   - Analyze WSS patterns across shapes
   - Identify high-risk regions

3. **Hyperparameter tuning**:
   - Use Optuna for systematic optimization
   - Find best loss weights

4. **Unified model**:
   - Train single model on all 6 geometries
   - Learn cross-geometry patterns

---

## 🎓 Understanding the Output

### What the Networks Learn

**Net_u, Net_v, Net_w**:
- Predict velocity field everywhere in 3D space
- Learn from WSS boundary constraints + physics equations

**Net_p**:
- Predicts pressure distribution
- Learns from CFD pressure data + Navier-Stokes

### What Gets Validated

**WSS Metrics**:
- How well predicted WSS matches CFD ground truth
- Primary indicator of model accuracy

**Physics Residuals**:
- How well Navier-Stokes equations are satisfied
- Should approach zero (perfect physics)

---

**Happy Training! 🚀**

For detailed information, see `README.md`
