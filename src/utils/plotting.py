"""
Generate publication-ready comparison plots: CFD vs PINN predictions.

Produces for each geometry & phase:
  - 3-panel plots (CFD | PINN | Absolute Error) with shared colorbar
    for WSS magnitude, WSS components, and pressure on XY, XZ, YZ planes.

Uses the FULL dataset for prediction. Coordinate centering is consistent
(full-data mean) between training and inference.

Usage:
  # Single geometry (best_model.pt):
  python -m src.utils.plotting --geom AD5

  # All completed geometries:
  python -m src.utils.plotting --all

  # Custom checkpoint:
  python -m src.utils.plotting --geom AD5 --checkpoint experiments/AD5/best_model.pt
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

from src.models.networks import create_taa_networks
from src.utils.geometry import compute_wall_normals_torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ── Geometry → config mapping ──────────────────────────────────────────────
GEOM_INFO = {
    "AS5": {
        "files": {"systolic": "5cm systolic.csv",      "diastolic": "5cm diastolic.csv"},
        "config": "configs/AS5_config.yaml",
    },
    "AD5": {
        "files": {"systolic": "5cm ASD systolic.csv",  "diastolic": "5cm ASD Diastolic.csv"},
        "config": "configs/AD5_config.yaml",
    },
    "PD5": {
        "files": {"systolic": "5cm ASU systolic.csv",  "diastolic": "5cm ASU Diastolic.csv"},
        "config": "configs/PD5_config.yaml",
    },
    "AS6": {
        "files": {"systolic": "6cm systolic.csv",      "diastolic": "6cm diastolic.csv"},
        "config": "configs/AS6_config.yaml",
    },
    "AD6": {
        "files": {"systolic": "6cm ASD Systolic.csv",  "diastolic": "6cm ASD diastolic.csv"},
        "config": "configs/AD6_config.yaml",
    },
    "PD6": {
        "files": {"systolic": "6cm ASU systolic.csv",  "diastolic": "6cm ASU Diastolic.csv"},
        "config": "configs/PD6_config.yaml",
    },
}

PLANES = [
    {"name": "xy", "xi": 0, "yi": 1, "xl": "X (m)", "yl": "Y (m)"},
    {"name": "xz", "xi": 0, "yi": 2, "xl": "X (m)", "yl": "Z (m)"},
    {"name": "yz", "xi": 1, "yi": 2, "xl": "Y (m)", "yl": "Z (m)"},
]


# ── Data loading (matches training exactly) ───────────────────────────────

def load_csv_data(filepath, subsample_factor=1):
    """Load a CSV the same way TAADataLoader does, returning raw + normalized."""
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    header_idx = None
    for i, line in enumerate(lines):
        if 'X [ m ]' in line:
            header_idx = i
            break

    df = pd.read_csv(filepath, skiprows=list(range(header_idx)), low_memory=False)
    df.columns = df.columns.str.strip()
    df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna().reset_index(drop=True)

    if subsample_factor > 1:
        df = df.iloc[::subsample_factor].reset_index(drop=True)

    return df


def prepare_data(df, L_ref, coord_scale=1.0):
    """Normalize coordinates matching TAADataLoader's two-step pipeline.

    Step 1: x_bar = (x - x_mean) / L_ref
    Step 2: x_std = x_bar / coord_scale  (brings coordinates to [-1, 1])

    coord_scale is read from the checkpoint's ref_scales and defaults to 1.0
    for backward compatibility with checkpoints trained before this fix.

    Pressure and WSS are returned raw; the caller converts using ref_scales.
    """
    coords = np.column_stack([
        df['X [ m ]'].values,
        df['Y [ m ]'].values,
        df['Z [ m ]'].values,
    ])
    coords_mean = coords.mean(axis=0)
    coords_norm = (coords - coords_mean) / L_ref / coord_scale

    pressure_raw = df['Pressure [ Pa ]'].values
    wss_x = df['Wall Shear X [ Pa ]'].values
    wss_y = df['Wall Shear Y [ Pa ]'].values
    wss_z = df['Wall Shear Z [ Pa ]'].values
    wss_mag = df['Wall Shear [ Pa ]'].values

    return {
        'coords_raw': coords,
        'coords_norm': coords_norm,
        'coords_mean': coords_mean,
        'pressure_raw': pressure_raw,
        'wss_magnitude_raw': wss_mag,
        'wss_x_raw': wss_x,
        'wss_y_raw': wss_y,
        'wss_z_raw': wss_z,
    }


# ── WSS computation via autograd (same maths as wss_loss.py) ──────────────

def compute_nut_from_network(networks, x, y, z, t_phase):
    """Compute predicted non-dimensional turbulent viscosity at given points.

    Returns nut_bar (non-dimensional) as a detached tensor.
    """
    inp = torch.cat([x, y, z, t_phase], dim=1)
    with torch.no_grad():
        nut = networks["nut"](inp).view(-1, 1)
    return nut


def compute_wss_from_networks(networks, x, y, z, t_phase, normals,
                              tau_ref, wss_std=1.0, pressure_std=1.0,
                              coord_scale=1.0):
    """Compute predicted WSS and pressure in physical units (Pa).

    The network was trained with:
      - Standardised coordinates:  x_std = x_bar / coord_scale  ∈ [-1, 1]
      - Standardised WSS targets:  tau_std = tau_bar / wss_std
      - Standardised pressure:     p_std = p_bar / pressure_std

    Autograd gives du_bar/d(x_std) = coord_scale · du_bar/d(x_bar).
    The physical stress tensor in x_bar space is:
        tau_bar_ij = (1/coord_scale) · (du_bar_i/dx_std_j + du_bar_j/dx_std_i)

    Full de-normalisation to physical units (Pa):
        WSS_Pa = tau_bar · tau_ref
               = (1/coord_scale) · du/dx_std · tau_ref

    Args:
        networks:     dict of u/v/w/p networks
        x, y, z:      standardised coords in [-1,1], shape (N,1)
        t_phase:      cardiac phase, shape (N,1)
        normals:      inward wall normals, shape (N,3)
        tau_ref:      viscous stress scale (Pa) = mu * U_ref / L_ref
        wss_std:      WSS standardisation factor (used for targets, not de-norm here)
        pressure_std: pressure standardisation factor
        coord_scale:  coordinate normalisation factor (x_bar = x_std * coord_scale)

    Returns:
        wss_x, wss_y, wss_z, wss_mag  (Pa),  p_model (standardised, needs * pressure_std * P_ref)
    """
    x = x.clone().detach().requires_grad_(True)
    y = y.clone().detach().requires_grad_(True)
    z = z.clone().detach().requires_grad_(True)

    inp = torch.cat([x, y, z, t_phase], dim=1)
    u = networks["u"](inp).view(-1, 1)
    v = networks["v"](inp).view(-1, 1)
    w = networks["w"](inp).view(-1, 1)
    p = networks["p"](inp).view(-1, 1)

    ones = torch.ones_like(u)
    grads = {}
    for name, field in [("u", u), ("v", v), ("w", w)]:
        for coord_name, coord in [("x", x), ("y", y), ("z", z)]:
            grads[f"{name}_{coord_name}"] = torch.autograd.grad(
                field, coord, grad_outputs=ones,
                create_graph=False, retain_graph=True,
            )[0]

    # Non-dimensional stress tensor in x_bar coordinates:
    #   tau_bar_ij = (1/coord_scale) · (du_bar_i/dx_std_j + du_bar_j/dx_std_i)
    inv_cs = 1.0 / coord_scale
    tau_xx = 2.0 * grads["u_x"] * inv_cs
    tau_yy = 2.0 * grads["v_y"] * inv_cs
    tau_zz = 2.0 * grads["w_z"] * inv_cs
    tau_xy = (grads["u_y"] + grads["v_x"]) * inv_cs
    tau_xz = (grads["u_z"] + grads["w_x"]) * inv_cs
    tau_yz = (grads["v_z"] + grads["w_y"]) * inv_cs

    nx, ny, nz = normals[:, 0:1], normals[:, 1:2], normals[:, 2:3]
    tx = tau_xx * nx + tau_xy * ny + tau_xz * nz
    ty = tau_xy * nx + tau_yy * ny + tau_yz * nz
    tz = tau_xz * nx + tau_yz * ny + tau_zz * nz
    t_dot_n = tx * nx + ty * ny + tz * nz

    # Non-dimensional WSS (tau_bar, in x_bar coordinates)
    wss_x_nd = tx - t_dot_n * nx
    wss_y_nd = ty - t_dot_n * ny
    wss_z_nd = tz - t_dot_n * nz

    # Convert tau_bar → physical Pa:  WSS_Pa = tau_bar * tau_ref
    wss_x = wss_x_nd * tau_ref
    wss_y = wss_y_nd * tau_ref
    wss_z = wss_z_nd * tau_ref
    wss_mag = torch.sqrt(wss_x ** 2 + wss_y ** 2 + wss_z ** 2 + 1e-12)

    return wss_x.detach(), wss_y.detach(), wss_z.detach(), wss_mag.detach(), p.detach()


# ── Publication-quality matplotlib defaults ───────────────────────────────

PUB_RC = {
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
}

FIELD_LABELS = {
    "WSS_magnitude": r"$|\tau_w|$ (Pa)",
    "WSS_x": r"$\tau_{w,x}$ (Pa)",
    "WSS_y": r"$\tau_{w,y}$ (Pa)",
    "WSS_z": r"$\tau_{w,z}$ (Pa)",
    "Pressure": r"$p$ (Pa)",
    "nut": r"$\bar{\nu}_t$",
}


def _apply_pub_style():
    """Apply publication rcParams (call before creating figures)."""
    plt.rcParams.update(PUB_RC)


# ── Plot helpers ──────────────────────────────────────────────────────────

def _scatter(ax, coords, vals, plane, s, cmap, vmin, vmax, title):
    sc = ax.scatter(
        coords[:, plane["xi"]], coords[:, plane["yi"]],
        c=vals, s=s, cmap=cmap, vmin=vmin, vmax=vmax,
        edgecolors="none", rasterized=True,
    )
    ax.set_xlabel(plane["xl"])
    ax.set_ylabel(plane["yl"])
    ax.set_title(title, fontsize=14)
    ax.set_aspect("equal", adjustable="box")
    return sc


def plot_field_comparison(coords, cfd_vals, pinn_vals, geom, phase, field_name,
                          out_dir, s=4, cmap="viridis", dpi=300,
                          clip_percentile=1.0):
    """3-panel per plane: CFD | PINN | Absolute Error."""
    _apply_pub_style()
    error = np.abs(pinn_vals - cfd_vals)

    # For WSS magnitude, floor at 0
    if "magnitude" in field_name.lower():
        vmin_floor = 0.0
    else:
        vmin_floor = None

    combined = np.concatenate([cfd_vals, pinn_vals])
    vmin = np.percentile(combined, clip_percentile)
    vmax = np.percentile(combined, 100.0 - clip_percentile)
    if vmin_floor is not None:
        vmin = max(vmin, vmin_floor)
    if np.isclose(vmin, vmax):
        vmin, vmax = float(combined.min()), float(combined.max())

    err_vmax = np.percentile(error, 95)
    cb_label = FIELD_LABELS.get(field_name, f"{field_name} (Pa)")

    saved = []
    panel_labels = ["(a)", "(b)", "(c)"]
    for plane in PLANES:
        fig, axes = plt.subplots(1, 3, figsize=(20, 5.5), constrained_layout=True)
        titles = ["CFD", "PINN", "Absolute Error"]
        sc = _scatter(axes[0], coords, cfd_vals, plane, s, cmap, vmin, vmax, titles[0])
        _scatter(axes[1], coords, pinn_vals, plane, s, cmap, vmin, vmax, titles[1])
        sc_err = _scatter(axes[2], coords, error, plane, s, "hot", 0, err_vmax, titles[2])

        for i, ax in enumerate(axes):
            ax.text(0.02, 0.96, panel_labels[i], transform=ax.transAxes,
                    fontsize=14, fontweight="bold", va="top")

        fig.colorbar(sc, ax=axes[:2].tolist(), shrink=0.85, label=cb_label)
        fig.colorbar(sc_err, ax=axes[2], shrink=0.85, label="Error (Pa)")

        p = out_dir / f"{geom}_{phase}_{field_name}_{plane['name']}.png"
        fig.savefig(p, dpi=dpi)
        plt.close(fig)
        saved.append(p)
    return saved


# ── Scatter / regression plots ────────────────────────────────────────────

def plot_scatter_regression(cfd_vals, pinn_vals, field_name, geom, phase,
                            out_dir, dpi=300):
    """PINN vs CFD scatter plot with regression line and R^2."""
    _apply_pub_style()

    mask = np.isfinite(cfd_vals) & np.isfinite(pinn_vals)
    x, y = cfd_vals[mask], pinn_vals[mask]

    corr = np.corrcoef(x, y)[0, 1]
    r2 = corr ** 2
    rel_l2 = np.linalg.norm(y - x) / (np.linalg.norm(x) + 1e-12)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(x, y, s=0.4, alpha=0.3, rasterized=True, color="steelblue")

    lo = min(x.min(), y.min())
    hi = max(x.max(), y.max())
    margin = 0.05 * (hi - lo)
    ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin],
            "k--", linewidth=1.0, label="Identity")
    ax.set_xlim(lo - margin, hi + margin)
    ax.set_ylim(lo - margin, hi + margin)

    cb_label = FIELD_LABELS.get(field_name, f"{field_name} (Pa)")
    ax.set_xlabel(f"CFD {cb_label}")
    ax.set_ylabel(f"PINN {cb_label}")
    ax.set_title(f"{geom} {phase} — {field_name.replace('_', ' ')}")
    ax.set_aspect("equal")

    textstr = f"$R^2 = {r2:.4f}$\nRel. $L_2 = {rel_l2:.4f}$\n$r = {corr:.4f}$"
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12,
            va="top", bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.8))

    p = out_dir / f"{geom}_{phase}_{field_name}_scatter.png"
    fig.savefig(p, dpi=dpi)
    plt.close(fig)
    return p


def plot_error_histogram(cfd_vals, pinn_vals, field_name, geom, phase,
                         out_dir, dpi=300):
    """Pointwise error distribution histogram."""
    _apply_pub_style()

    error = pinn_vals - cfd_vals
    abs_error = np.abs(error)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    cb_label = FIELD_LABELS.get(field_name, f"{field_name} (Pa)")

    axes[0].hist(error, bins=100, color="steelblue", edgecolor="none", alpha=0.8)
    axes[0].axvline(0, color="k", linestyle="--", linewidth=0.8)
    axes[0].set_xlabel(f"Error ({cb_label.split('(')[-1]}")
    axes[0].set_ylabel("Count")
    axes[0].set_title("(a) Signed Error Distribution")
    mean_err = np.mean(error)
    axes[0].axvline(mean_err, color="red", linestyle="-", linewidth=1.0)
    axes[0].text(0.95, 0.95, f"Mean = {mean_err:.3f}\nStd = {np.std(error):.3f}",
                 transform=axes[0].transAxes, fontsize=11, va="top", ha="right",
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    axes[1].hist(abs_error, bins=100, color="coral", edgecolor="none", alpha=0.8)
    axes[1].set_xlabel(f"|Error| ({cb_label.split('(')[-1]}")
    axes[1].set_ylabel("Count")
    axes[1].set_title("(b) Absolute Error Distribution")
    p50 = np.percentile(abs_error, 50)
    p95 = np.percentile(abs_error, 95)
    axes[1].axvline(p50, color="blue", linestyle="--", linewidth=1.0, label=f"P50={p50:.3f}")
    axes[1].axvline(p95, color="red", linestyle="--", linewidth=1.0, label=f"P95={p95:.3f}")
    axes[1].legend()

    fig.suptitle(f"{geom} {phase} — {field_name.replace('_', ' ')}", fontsize=14)
    p = out_dir / f"{geom}_{phase}_{field_name}_error_hist.png"
    fig.savefig(p, dpi=dpi)
    plt.close(fig)
    return p


# ── Publication loss curves (individual PDFs) ─────────────────────────────

def generate_publication_loss_plots(geom, out_dir=None):
    """Generate individual publication-quality loss curve plots from loss_history.csv."""
    _apply_pub_style()
    exp_dir = PROJECT_ROOT / "experiments" / geom
    csv_path = exp_dir / "loss_history.csv"
    if not csv_path.exists():
        print(f"  [SKIP] No loss_history.csv for {geom}")
        return

    if out_dir is None:
        out_dir = exp_dir / "figures"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    epochs = df["epoch"].values

    # 1) Total loss
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.semilogy(epochs, df["total"].values, color="steelblue", linewidth=1.0)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(r"$\mathcal{L}_{\mathrm{total}}$")
    ax.set_title(f"{geom} — Total Training Loss")
    ax.grid(True, alpha=0.3)
    fig.savefig(out_dir / f"{geom}_total_loss.pdf")
    plt.close(fig)

    # 2) Component losses
    fig, ax = plt.subplots(figsize=(7, 4.5))
    components = [
        ("wss", r"$\mathcal{L}_{\mathrm{WSS}}$", "steelblue"),
        ("physics", r"$\mathcal{L}_{\mathrm{physics}}$", "darkorange"),
        ("bc_noslip", r"$\mathcal{L}_{\mathrm{BC}}$", "forestgreen"),
        ("pressure", r"$\mathcal{L}_{\mathrm{pressure}}$", "crimson"),
    ]
    for key, label, color in components:
        if key in df.columns:
            ax.semilogy(epochs, df[key].values, label=label, color=color, linewidth=1.0)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"{geom} — Component Losses")
    ax.legend(framealpha=0.9)
    ax.grid(True, alpha=0.3)
    fig.savefig(out_dir / f"{geom}_component_losses.pdf")
    plt.close(fig)

    # 3) Physics residuals
    fig, ax = plt.subplots(figsize=(7, 4.5))
    residuals = [
        ("res_mom_x", r"$r_{u}$", "steelblue"),
        ("res_mom_y", r"$r_{v}$", "darkorange"),
        ("res_mom_z", r"$r_{w}$", "forestgreen"),
        ("res_cont", r"$r_{\nabla \cdot \mathbf{u}}$", "crimson"),
    ]
    for key, label, color in residuals:
        if key in df.columns:
            ax.semilogy(epochs, df[key].values, label=label, color=color, linewidth=1.0)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Residual (mean absolute)")
    ax.set_title(f"{geom} — N-S Residuals")
    ax.legend(framealpha=0.9)
    ax.grid(True, alpha=0.3)
    fig.savefig(out_dir / f"{geom}_physics_residuals.pdf")
    plt.close(fig)

    # 4) Learning rate
    if "lr" in df.columns:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.semilogy(epochs, df["lr"].values, color="steelblue", linewidth=1.0)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Learning Rate")
        ax.set_title(f"{geom} — Learning Rate Schedule")
        ax.grid(True, alpha=0.3)
        fig.savefig(out_dir / f"{geom}_learning_rate.pdf")
        plt.close(fig)

    # 5) Adaptive weights (if present)
    aw_cols = [c for c in df.columns if c.startswith("aw_")]
    if aw_cols:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        aw_labels = {"aw_wss": r"$\lambda_{\mathrm{WSS}}$",
                     "aw_physics": r"$\lambda_{\mathrm{physics}}$",
                     "aw_bc_noslip": r"$\lambda_{\mathrm{BC}}$",
                     "aw_pressure": r"$\lambda_{\mathrm{pressure}}$"}
        for col in aw_cols:
            vals = df[col].dropna()
            if len(vals) > 0:
                ax.semilogy(df.loc[vals.index, "epoch"], vals,
                            label=aw_labels.get(col, col), linewidth=1.0)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Adaptive Weight")
        ax.set_title(f"{geom} — Adaptive Loss Weights")
        ax.legend(framealpha=0.9)
        ax.grid(True, alpha=0.3)
        fig.savefig(out_dir / f"{geom}_adaptive_weights.pdf")
        plt.close(fig)

    print(f"  Publication loss plots saved to {out_dir}")


# ── Cross-geometry summary ────────────────────────────────────────────────

def generate_summary_table_and_charts(out_dir=None):
    """Read all evaluation_metrics.csv files and produce a summary CSV + bar charts."""
    _apply_pub_style()
    if out_dir is None:
        out_dir = PROJECT_ROOT / "experiments"
    out_dir = Path(out_dir)

    rows = []
    for geom in GEOM_INFO:
        csv_path = PROJECT_ROOT / "experiments" / geom / "evaluation_metrics.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            rows.append({
                "geometry": geom,
                "phase": row["phase"],
                "epoch": int(row["epoch"]),
                "wss_rel_l2": row["wss_relative_l2"],
                "wss_correlation": row["wss_correlation"],
                "wss_mae": row["wss_mae"],
                "wss_rmse": row["wss_rmse"],
                "loss_total": row["loss_total"],
                "loss_wss": row["loss_wss"],
                "loss_physics": row["loss_physics"],
                "res_mom_x": row["residual_momentum_x"],
                "res_mom_y": row["residual_momentum_y"],
                "res_mom_z": row["residual_momentum_z"],
                "res_continuity": row["residual_continuity"],
            })

    if not rows:
        print("  No evaluation metrics found.")
        return

    summary = pd.DataFrame(rows)
    summary_path = out_dir / "summary_metrics.csv"
    summary.to_csv(summary_path, index=False, float_format="%.6f")
    print(f"  Summary table saved: {summary_path}")

    # Bar chart: WSS Relative L2 by geometry and phase
    geoms = summary["geometry"].unique()
    phases = summary["phase"].unique()
    x = np.arange(len(geoms))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)

    for i, metric, ylabel, title in [
        (0, "wss_rel_l2", r"Relative $L_2$ Error", "(a) WSS Relative $L_2$"),
        (1, "wss_correlation", "Pearson Correlation", "(b) WSS Correlation"),
    ]:
        ax = axes[i]
        for j, phase in enumerate(phases):
            subset = summary[summary["phase"] == phase]
            vals = []
            for g in geoms:
                row = subset[subset["geometry"] == g]
                vals.append(row[metric].values[0] if len(row) > 0 else 0)
            offset = (j - 0.5) * width
            bars = ax.bar(x + offset, vals, width, label=phase.capitalize(),
                          alpha=0.85)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f"{v:.3f}", ha="center", va="bottom", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(geoms)
        ax.set_xlabel("Geometry")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.2, axis="y")

    fig.savefig(out_dir / "summary_bar_chart.pdf")
    fig.savefig(out_dir / "summary_bar_chart.png", dpi=300)
    plt.close(fig)
    print(f"  Summary bar chart saved: {out_dir / 'summary_bar_chart.pdf'}")


# ── Main logic ────────────────────────────────────────────────────────────

def process_geometry(geom, checkpoint_path=None, device="cuda"):
    """Generate all comparison plots for one geometry using the full dataset."""
    info = GEOM_INFO[geom]
    exp_dir = PROJECT_ROOT / "experiments" / geom
    config_path = PROJECT_ROOT / info["config"]

    if checkpoint_path is None:
        for name in ["best_model.pt", "final_model.pt"]:
            candidate = exp_dir / name
            if candidate.exists():
                checkpoint_path = str(candidate)
                break
    if checkpoint_path is None:
        print(f"  [SKIP] No checkpoint found for {geom}")
        return

    print(f"\n{'='*60}")
    print(f"  Processing {geom}  |  checkpoint: {Path(checkpoint_path).name}")
    print(f"{'='*60}")

    # Load checkpoint & create networks
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model_cfg = cfg["model"]
    norm = cfg["data"]["normalization"]
    nut_cfg = model_cfg.get("nut", {})
    networks = create_taa_networks(
        input_dim=model_cfg.get("input_dim", 4),
        hidden_dim=model_cfg.get("hidden_dim", 256),
        num_layers=model_cfg.get("num_layers", 10),
        num_frequencies=model_cfg.get("num_frequencies", 32),
        fourier_scale=model_cfg.get("fourier_scale", 10.0),
        use_fourier=model_cfg.get("use_fourier", True),
        nut_hidden_dim=nut_cfg.get("hidden_dim", 64),
        nut_num_layers=nut_cfg.get("num_layers", 4),
        device=device,
    )
    for name, net in networks.items():
        if name in ckpt["networks"]:
            net.load_state_dict(ckpt["networks"][name])
        else:
            print(f"  Warning: '{name}' not in checkpoint; using fresh init.")
        net.eval()

    L = norm["length_scale"]

    # Extract non-dimensional reference scales from checkpoint
    ref_scales = ckpt.get("ref_scales", {})
    P_ref = ref_scales.get("P_ref", 1.0)
    tau_ref = ref_scales.get("tau_ref", 1.0)
    # Standardisation factors (default to 1.0 for backward compatibility with
    # checkpoints trained before the standardisation fix was applied)
    wss_std = ref_scales.get("wss_std", 1.0)
    pressure_std = ref_scales.get("pressure_std", 1.0)
    coord_scale = ref_scales.get("coord_scale", 1.0)
    train_subsample = cfg["data"].get("subsample_factor", 1)

    # Read the geometry config for normal estimation params
    if config_path.exists():
        with open(config_path) as f:
            geom_cfg = yaml.safe_load(f)
        normal_radius = geom_cfg["geometry"]["normal_estimation"]["radius"]
        normal_max_nn = geom_cfg["geometry"]["normal_estimation"]["max_nn"]
    else:
        normal_radius = 0.01
        normal_max_nn = 30

    out_dir = exp_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    for phase, filename in info["files"].items():
        print(f"  Phase: {phase}  ({filename})")
        filepath = str(PROJECT_ROOT / "data" / filename)

        # Load the FULL dataset — coordinate centering now uses the full-data
        # mean in TAADataLoader (consistent regardless of subsample_factor),
        # so a single load at subsample=1 gives correctly centred coords.
        df_full = load_csv_data(filepath, subsample_factor=1)
        full_data = prepare_data(df_full, L, coord_scale=coord_scale)
        coords_full_raw = full_data['coords_raw']
        coords_full_norm = full_data['coords_norm']

        # Full CFD ground truth
        cfd_pressure = full_data['pressure_raw']
        cfd_wss_mag  = full_data['wss_magnitude_raw']
        cfd_wss_x    = full_data['wss_x_raw']
        cfd_wss_y    = full_data['wss_y_raw']
        cfd_wss_z    = full_data['wss_z_raw']

        n_pts = coords_full_norm.shape[0]
        print(f"    {n_pts} points (full dataset, centred with training mean)")

        # Prepare tensors
        x = torch.tensor(coords_full_norm[:, 0:1], dtype=torch.float32, device=device)
        y = torch.tensor(coords_full_norm[:, 1:2], dtype=torch.float32, device=device)
        z = torch.tensor(coords_full_norm[:, 2:3], dtype=torch.float32, device=device)
        phase_val = 1.0 if phase == "systolic" else 0.0
        t = torch.full((n_pts, 1), phase_val, dtype=torch.float32, device=device)

        # Compute wall normals
        normals = compute_wall_normals_torch(
            x, y, z, radius=normal_radius, max_nn=normal_max_nn, device=device,
        )

        # Compute PINN predictions (batched)
        # Model outputs are non-dimensional; convert back to Pa for plotting
        batch = 4000
        wss_x_list, wss_y_list, wss_z_list, wss_mag_list, p_list = [], [], [], [], []
        for i in range(0, n_pts, batch):
            j = min(i + batch, n_pts)
            wx, wy, wz, wm, pp = compute_wss_from_networks(
                networks, x[i:j], y[i:j], z[i:j], t[i:j], normals[i:j],
                tau_ref=tau_ref, wss_std=wss_std, pressure_std=pressure_std,
                coord_scale=coord_scale,
            )
            wss_x_list.append(wx.cpu())
            wss_y_list.append(wy.cpu())
            wss_z_list.append(wz.cpu())
            wss_mag_list.append(wm.cpu())
            p_list.append(pp.cpu())

        pinn_wss_mag = torch.cat(wss_mag_list).numpy().flatten()
        pinn_wss_x   = torch.cat(wss_x_list).numpy().flatten()
        pinn_wss_y   = torch.cat(wss_y_list).numpy().flatten()
        pinn_wss_z   = torch.cat(wss_z_list).numpy().flatten()
        # p_model is in standardised non-dim units; full inverse: * pressure_std * P_ref
        pinn_pressure = torch.cat(p_list).numpy().flatten() * pressure_std * P_ref

        # Print quick metrics
        for fname, cfd, pinn in [
            ("WSS_mag", cfd_wss_mag, pinn_wss_mag),
            ("Pressure", cfd_pressure, pinn_pressure),
        ]:
            rel_l2 = np.linalg.norm(pinn - cfd) / (np.linalg.norm(cfd) + 1e-12)
            corr = np.corrcoef(cfd.flatten(), pinn.flatten())[0, 1]
            print(f"    {fname}: Rel L2={rel_l2:.4f}, Corr={corr:.4f}, "
                  f"CFD=[{cfd.min():.2f},{cfd.max():.2f}], "
                  f"PINN=[{pinn.min():.2f},{pinn.max():.2f}]")

        # 1) WSS magnitude comparison (3 planes)
        print("    Plotting WSS magnitude...")
        plot_field_comparison(coords_full_raw, cfd_wss_mag, pinn_wss_mag,
                              geom, phase, "WSS_magnitude", out_dir)

        # 2) WSS components (3 planes each)
        for ci, comp_name in enumerate(["WSS_x", "WSS_y", "WSS_z"]):
            print(f"    Plotting {comp_name}...")
            pinn_comp = [pinn_wss_x, pinn_wss_y, pinn_wss_z][ci]
            cfd_comp = [cfd_wss_x, cfd_wss_y, cfd_wss_z][ci]
            plot_field_comparison(coords_full_raw, cfd_comp, pinn_comp,
                                  geom, phase, comp_name, out_dir, cmap="coolwarm")

        # 3) Pressure comparison (3 planes)
        print("    Plotting pressure...")
        plot_field_comparison(coords_full_raw, cfd_pressure, pinn_pressure,
                              geom, phase, "Pressure", out_dir)

        # 4) Scatter/regression plots
        print("    Plotting scatter/regression...")
        plot_scatter_regression(cfd_wss_mag, pinn_wss_mag, "WSS_magnitude",
                                geom, phase, out_dir)
        plot_scatter_regression(cfd_pressure, pinn_pressure, "Pressure",
                                geom, phase, out_dir)

        # 5) Error histograms
        print("    Plotting error histograms...")
        plot_error_histogram(cfd_wss_mag, pinn_wss_mag, "WSS_magnitude",
                             geom, phase, out_dir)
        plot_error_histogram(cfd_pressure, pinn_pressure, "Pressure",
                             geom, phase, out_dir)

        # 6) Turbulent viscosity field (if nut network present)
        if "nut" in networks:
            print("    Plotting turbulent viscosity field...")
            nut_list = []
            for i in range(0, n_pts, batch):
                j = min(i + batch, n_pts)
                nut_batch = compute_nut_from_network(
                    networks, x[i:j], y[i:j], z[i:j], t[i:j])
                nut_list.append(nut_batch.cpu())
            pinn_nut = torch.cat(nut_list).numpy().flatten()
            print(f"    nu_t_bar: mean={pinn_nut.mean():.6f}, "
                  f"max={pinn_nut.max():.6f}, min={pinn_nut.min():.6f}")

            for plane in PLANES:
                _apply_pub_style()
                fig, ax = plt.subplots(1, 1, figsize=(8, 6), constrained_layout=True)
                vmin_nut = 0.0
                vmax_nut = np.percentile(pinn_nut, 99)
                if np.isclose(vmax_nut, 0.0):
                    vmax_nut = pinn_nut.max() + 1e-8
                sc = _scatter(ax, coords_full_raw, pinn_nut, plane, s=4,
                              cmap="inferno", vmin=vmin_nut, vmax=vmax_nut,
                              title=f"{geom} {phase} — Predicted $\\bar{{\\nu}}_t$")
                fig.colorbar(sc, ax=ax, shrink=0.85, label=r"$\bar{\nu}_t$")
                p_out = out_dir / f"{geom}_{phase}_nut_{plane['name']}.png"
                fig.savefig(p_out, dpi=300)
                plt.close(fig)

        print(f"    Done ({phase})")


# ── CLI ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate CFD vs PINN comparison plots")
    parser.add_argument("--geom", type=str, default=None,
                        help="Geometry code (AS5, AD5, PD5, AS6, AD6, PD6)")
    parser.add_argument("--all", action="store_true",
                        help="Process all geometries with available checkpoints")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to specific checkpoint (overrides auto-detection)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--loss-plots", action="store_true",
                        help="Generate publication-quality loss curve plots (PDF)")
    parser.add_argument("--summary", action="store_true",
                        help="Generate cross-geometry summary table and bar charts")
    args = parser.parse_args()

    geoms = []
    if args.all:
        geoms = list(GEOM_INFO.keys())
    elif args.geom:
        geoms = [args.geom.upper()]

    if args.summary:
        print("\n=== Generating cross-geometry summary ===")
        generate_summary_table_and_charts()

    if args.loss_plots:
        targets = geoms if geoms else list(GEOM_INFO.keys())
        for geom in targets:
            print(f"\n=== Loss plots: {geom} ===")
            generate_publication_loss_plots(geom)

    if geoms and not args.loss_plots and not args.summary:
        for geom in geoms:
            process_geometry(geom, args.checkpoint, args.device)
    elif geoms and (args.loss_plots or args.summary):
        pass  # already handled above
    elif not geoms and not args.loss_plots and not args.summary:
        parser.error("Specify --geom GEOM, --all, --loss-plots, or --summary")

    print("\nDone!")


if __name__ == "__main__":
    main()
