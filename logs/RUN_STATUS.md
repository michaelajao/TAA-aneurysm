# Training run status (as of check)

## Completed runs (all six)

| Config | Status | Notes |
|--------|--------|--------|
| AS5 | Done | Rerun with patience=1000 in progress (was 3000) |
| AS6 | Done | Rerun with patience=1000 in progress (was 3000) |
| AD5 | Done | patience=1000 |
| AD6 | Done | patience=1000 |
| PD5 | Done | patience=1000 |
| PD6 | Done | patience=1000 |

**Early stopping:** All configs use `patience: 1000` now. AS5 and AS6 were originally run with patience=3000; reruns with patience=1000 are launched so results are consistent.

## Reruns (patience=1000)

AS5 and AS6 are being rerun with the same early-stopping patience (1000) as the other configs. Logs: `logs/AS5_patience1000_*.log`, `logs/AS6_patience1000_*.log`.

## When a run completes

The trainer writes to `experiments/<GEOM>/`:

- `loss_curves.png` – loss history
- `evaluation_metrics.csv` – WSS and other metrics per phase
- `best_model.pt` – best checkpoint
- CFD vs PINN comparison plots (if generation succeeds)

Check GPU: `nvidia-smi`
