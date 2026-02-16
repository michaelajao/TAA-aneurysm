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
  python -m src.utils.plotting --geom AD5 --checkpoint experiments/AD5_adaptive_physics/best_model.pt
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


def prepare_data(df, L_ref):
    """Normalize coordinates as TAADataLoader does: center then divide by L_ref.

    Pressure and WSS are returned raw; the caller converts using ref_scales.
    """
    coords = np.column_stack([
        df['X [ m ]'].values,
        df['Y [ m ]'].values,
        df['Z [ m ]'].values,
    ])
    coords_mean = coords.mean(axis=0)
    coords_norm = (coords - coords_mean) / L_ref

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

def compute_wss_from_networks(networks, x, y, z, t_phase, normals, tau_ref):
    """Compute predicted WSS in Pa using autograd, matching the non-dimensional training loss.

    The model outputs non-dimensional velocity/pressure.  The non-dimensional
    WSS (viscous-scaled) is tau_bar_ij = du_bar_i/dx_bar_j + du_bar_j/dx_bar_i.
    Multiply by tau_ref to recover physical units (Pa).
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

    # Non-dimensional stress tensor (viscous scaling: mu cancels)
    tau_xx = 2.0 * grads["u_x"]
    tau_yy = 2.0 * grads["v_y"]
    tau_zz = 2.0 * grads["w_z"]
    tau_xy = grads["u_y"] + grads["v_x"]
    tau_xz = grads["u_z"] + grads["w_x"]
    tau_yz = grads["v_z"] + grads["w_y"]

    nx, ny, nz = normals[:, 0:1], normals[:, 1:2], normals[:, 2:3]
    tx = tau_xx * nx + tau_xy * ny + tau_xz * nz
    ty = tau_xy * nx + tau_yy * ny + tau_yz * nz
    tz = tau_xz * nx + tau_yz * ny + tau_zz * nz
    t_dot_n = tx * nx + ty * ny + tz * nz

    # Non-dimensional WSS (viscous-scaled)
    wss_x_nd = tx - t_dot_n * nx
    wss_y_nd = ty - t_dot_n * ny
    wss_z_nd = tz - t_dot_n * nz

    # Convert to physical units (Pa)
    wss_x = wss_x_nd * tau_ref
    wss_y = wss_y_nd * tau_ref
    wss_z = wss_z_nd * tau_ref
    wss_mag = torch.sqrt(wss_x ** 2 + wss_y ** 2 + wss_z ** 2 + 1e-12)

    return wss_x.detach(), wss_y.detach(), wss_z.detach(), wss_mag.detach(), p.detach()


# ── Plot helpers ──────────────────────────────────────────────────────────

def _scatter(ax, coords, vals, plane, s, cmap, vmin, vmax, title):
    sc = ax.scatter(
        coords[:, plane["xi"]], coords[:, plane["yi"]],
        c=vals, s=s, cmap=cmap, vmin=vmin, vmax=vmax,
        edgecolors="none", rasterized=True,
    )
    ax.set_xlabel(plane["xl"])
    ax.set_ylabel(plane["yl"])
    ax.set_title(title, fontsize=11)
    ax.set_aspect("equal", adjustable="box")
    return sc


def plot_field_comparison(coords, cfd_vals, pinn_vals, geom, phase, field_name,
                          out_dir, s=4, cmap="viridis", dpi=300,
                          clip_percentile=1.0):
    """3-panel per plane: CFD | PINN | Absolute Error (no suptitle)."""
    error = np.abs(pinn_vals - cfd_vals)

    # Shared color range for CFD & PINN panels
    combined = np.concatenate([cfd_vals, pinn_vals])
    vmin = np.percentile(combined, clip_percentile)
    vmax = np.percentile(combined, 100.0 - clip_percentile)
    if np.isclose(vmin, vmax):
        vmin, vmax = float(combined.min()), float(combined.max())

    # Error color range
    err_vmax = np.percentile(error, 95)

    saved = []
    for plane in PLANES:
        fig, axes = plt.subplots(1, 3, figsize=(20, 5), constrained_layout=True)
        sc = _scatter(axes[0], coords, cfd_vals, plane, s, cmap, vmin, vmax, "CFD")
        _scatter(axes[1], coords, pinn_vals, plane, s, cmap, vmin, vmax, "PINN")
        sc_err = _scatter(axes[2], coords, error, plane, s, "hot", 0, err_vmax,
                          "Absolute Error")

        fig.colorbar(sc, ax=axes[:2].tolist(), shrink=0.85, label=f"{field_name} (Pa)")
        fig.colorbar(sc_err, ax=axes[2], shrink=0.85, label="Error (Pa)")

        p = out_dir / f"{geom}_{phase}_{field_name}_{plane['name']}.png"
        fig.savefig(p, dpi=dpi)
        plt.close(fig)
        saved.append(p)
    return saved


# ── Main logic ────────────────────────────────────────────────────────────

def process_geometry(geom, checkpoint_path=None, device="cuda"):
    """Generate all comparison plots for one geometry using the full dataset."""
    info = GEOM_INFO[geom]
    exp_dir = PROJECT_ROOT / "experiments" / f"{geom}_adaptive_physics"
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
    networks = create_taa_networks(
        input_dim=model_cfg.get("input_dim", 4),
        hidden_dim=model_cfg.get("hidden_dim", 256),
        num_layers=model_cfg.get("num_layers", 10),
        num_frequencies=model_cfg.get("num_frequencies", 32),
        fourier_scale=model_cfg.get("fourier_scale", 10.0),
        use_fourier=model_cfg.get("use_fourier", True),
        device=device,
    )
    for name, net in networks.items():
        net.load_state_dict(ckpt["networks"][name])
        net.eval()

    L = norm["length_scale"]

    # Extract non-dimensional reference scales from checkpoint
    ref_scales = ckpt.get("ref_scales", {})
    P_ref = ref_scales.get("P_ref", 1.0)
    tau_ref = ref_scales.get("tau_ref", 1.0)
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

    out_dir = PROJECT_ROOT / "experiments" / "figures" / geom
    out_dir.mkdir(parents=True, exist_ok=True)

    for phase, filename in info["files"].items():
        print(f"  Phase: {phase}  ({filename})")
        filepath = str(PROJECT_ROOT / "data" / filename)

        # Load the FULL dataset — coordinate centering now uses the full-data
        # mean in TAADataLoader (consistent regardless of subsample_factor),
        # so a single load at subsample=1 gives correctly centred coords.
        df_full = load_csv_data(filepath, subsample_factor=1)
        full_data = prepare_data(df_full, L)
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
                networks, x[i:j], y[i:j], z[i:j], t[i:j], normals[i:j], tau_ref,
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
        pinn_pressure = torch.cat(p_list).numpy().flatten() * P_ref

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
    args = parser.parse_args()

    if args.all:
        geoms = list(GEOM_INFO.keys())
    elif args.geom:
        geoms = [args.geom.upper()]
    else:
        parser.error("Specify --geom GEOM or --all")

    for geom in geoms:
        process_geometry(geom, args.checkpoint, args.device)

    print("\nDone!")


if __name__ == "__main__":
    main()
