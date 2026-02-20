# Training run status (as of check)

## Current runs

| Config | GPU | Log file | Status |
|--------|-----|----------|--------|
| AS5 | 0 | logs/AS5_20260218_232651.log | Running (~12%, epoch ~1180/10000) |
| AS6 | 1 | logs/AS6_20260219_001759.log | Running (just started) |

## AS5 – Epoch 1000 evaluation (interim)

- **WSS**: Correlation **0.98** (systolic), **0.97** (diastolic); Relative L2 0.19 / 0.25 — **good**.
- **Physics**: Momentum residuals ~170–207, continuity ~143–151 (still high; typical for mid-training).
- **nu_t**: Mean ~0.002, max ~0.31–0.37 — **not collapsed**, learning.
- **Carreau-Yasuda μ_ratio**: Mean ~1.02, max ~1.09 — active.

Final metrics, figures, and comparison plots are written only when training **completes** (see below).

## When a run completes

The trainer will write to `experiments/<GEOM>/`:

- `loss_curves.png` – loss history
- `evaluation_metrics.csv` – WSS and other metrics per phase
- `best_model.pt` – best checkpoint (already updated during training)
- CFD vs PINN comparison plots (if generation succeeds)

## Run the remaining configs (AD5, AD6, PD5, PD6)

When a GPU is free, start one of these (use the free GPU index for `CUDA_VISIBLE_DEVICES`):

```bash
cd /home/olarinoyem/Project/TAA-aneurysm
source ~/miniconda3/etc/profile.d/conda.sh && conda activate deep_tf

# Example: AD5 on GPU 0 (when AS5 has finished)
LOGFILE="logs/AD5_$(date +%Y%m%d_%H%M%S).log"
CUDA_VISIBLE_DEVICES=0 nohup python -u -m src.training.trainer --config configs/AD5_config.yaml > "$LOGFILE" 2>&1 &
echo "Log: $LOGFILE"

# AD6 on GPU 1 (when AS6 has finished)
LOGFILE="logs/AD6_$(date +%Y%m%d_%H%M%S).log"
CUDA_VISIBLE_DEVICES=1 nohup python -u -m src.training.trainer --config configs/AD6_config.yaml > "$LOGFILE" 2>&1 &

# PD5 (use whichever GPU is free)
CUDA_VISIBLE_DEVICES=0 nohup python -u -m src.training.trainer --config configs/PD5_config.yaml > "logs/PD5_$(date +%Y%m%d_%H%M%S).log" 2>&1 &

# PD6 (use whichever GPU is free)
CUDA_VISIBLE_DEVICES=1 nohup python -u -m src.training.trainer --config configs/PD6_config.yaml > "logs/PD6_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
```

Check which GPU is free: `nvidia-smi`
