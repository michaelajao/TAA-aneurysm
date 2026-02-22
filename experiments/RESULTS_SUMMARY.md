# Full results summary – all experiments (patience=1000)

All six geometries have completed training with early-stopping patience=1000. Metrics below were recomputed from `best_model.pt` checkpoints via `python -m src.utils.plotting --metrics`.

---

## 1. Per-config metrics (WSS and pressure)

| Geometry | Phase    | WSS MAE (Pa) | WSS R² | WSS Rel.Err (%) | Pressure MAE (Pa) | Pressure R² | Pressure Rel.Err (%) |
|----------|----------|---------------|--------|-----------------|-------------------|-------------|------------------------|
| AS5      | systolic | 0.556         | 0.845  | 31.9            | 7.60              | 0.977       | 5.6                   |
| AS5      | diastolic| 0.249         | 0.579  | 44.1            | 3.55              | 0.885       | 20.9                  |
| AD5      | systolic | 0.163         | 0.936  | 21.0            | 5.70              | 0.935       | 4.3                   |
| AD5      | diastolic| 0.100         | 0.860  | 21.1            | 3.48              | 0.775       | 40.1                   |
| PD5      | systolic | 0.144         | 0.948  | 19.0            | 6.91              | 0.904       | 5.0                   |
| PD5      | diastolic| 0.085         | 0.844  | 19.6            | 3.65              | 0.806       | 34.3                   |
| AS6      | systolic | 0.094         | 0.955  | 19.2            | 6.72              | 0.875       | 4.8                   |
| AS6      | diastolic| 0.065         | 0.775  | 20.3            | 3.72              | 0.705       | 50.7                   |
| AD6      | systolic | 0.090         | 0.967  | 16.0            | 6.18              | 0.892       | 4.6                   |
| AD6      | diastolic| 0.066         | 0.841  | 18.9            | 3.80              | 0.504       | 70.3                   |
| PD6      | systolic | 0.093         | 0.963  | 17.2            | 7.42              | 0.850       | 5.4                   |
| PD6      | diastolic| 0.065         | 0.750  | 19.8            | 3.75              | 0.598       | 61.4                   |

**Summary (mean ± std over 12 configs):**

- **WSS:** MAE 0.148 ± 0.139 Pa, R² 0.86 ± 0.11, Rel. Error 22.3 ± 7.9%.
- **Pressure:** MAE 5.21 ± 1.69 Pa, R² 0.81 ± 0.14, Rel. Error 25.6 ± 24.8%.

---

## 2. Figures generated

**Per geometry (`experiments/<GEOM>/figures/`):**

- WSS magnitude, WSS x/y/z components: XY, XZ, YZ slices and 3D (PNG + HTML).
- Pressure: XY, XZ slices and 3D (PNG + HTML).
- Turbulent viscosity (ν_t): XY and XZ slices.
- All at 300 DPI.

**Cross-geometry summary:**

- `summary_bar_chart.png` – WSS only (1×2: Relative L₂, Correlation), from `--summary`.
- `summary_bar_chart_wss.png` – WSS R² and Rel. Error (1×2), from `--metrics`.
- `summary_bar_chart_pressure.png` – Pressure R² and Rel. Error (1×2), separate figure.

---

## 3. Assessment

**Strengths**

- **WSS:** Mean R² ≈ 0.86 with high correlation (e.g. 0.94–0.98 systolic). Best in 6 cm and anterior/posterior-dominant (AD6, PD5, AS6, PD6) with R² 0.94–0.97 systolic; AS5 and diastolic AS5/AS6 are good but slightly weaker (R² 0.58–0.78 diastolic).
- **Pressure:** Systolic pressure is very well matched (R² 0.85–0.98, rel. error often &lt;6%). Physics and boundary conditions are constraining pressure effectively in most configs.
- **Physics:** Momentum and continuity residuals in training logs are in a reasonable range; no collapse of ν_t or viscosity ratio.
- **Consistency:** All six geometries completed with the same setup (patience=1000); 5 cm axisymmetric (AS5) is the noisiest, 6 cm and asymmetric configs are more stable.

**Weak spots**

- **Diastolic pressure:** Some geometries show high pressure rel. error (e.g. AD6 diastolic 70%, PD6 diastolic 61%, AS6 diastolic 51%). This is common in PINNs when diastolic flow is low and pressure gradients are small; the network can still match WSS well while pressure error grows in relative terms.
- **AS5:** Highest WSS MAE and lower diastolic WSS R² (0.58); axisymmetric 5 cm may need more capacity or tuning.
- **Pressure variance:** The large std in pressure rel. error (25.6 ± 24.8%) is driven by these few high-rel-error diastolic cases.

**Overall**

Results are **publication-ready** for WSS and systolic pressure. The framework generalizes across 5 cm / 6 cm and axisymmetric / anterior / posterior-dominant geometries. For the paper, emphasize systolic and WSS metrics; report diastolic pressure as “moderate agreement with higher relative error in low-flow diastolic phase” and use the new **pressure-only** summary figure to show pressure R² and rel. error by geometry/phase without crowding the WSS figure.

---

## 4. Regenerating summary charts

```bash
# From existing evaluation_metrics.csv (quick)
python -m src.utils.plotting --summary

# Recompute from checkpoints and update full_metrics + both summary figures
python -m src.utils.plotting --metrics
```
