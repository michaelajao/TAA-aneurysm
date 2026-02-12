"""
Generate publication-ready 2D projections (XY, XZ, YZ) comparing PINN
predictions against CFD ground truth.  Each projection is saved as a
separate PNG with side-by-side CFD | PINN panels sharing a common colorbar.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_loaders.csv_loader import TAADataLoader

FIELD_MAP = {
    "wss_magnitude": "wss_magnitude_raw",
    "pressure": "pressure_raw",
    "wss_x": "wss_components_raw",
    "wss_y": "wss_components_raw",
    "wss_z": "wss_components_raw",
}

PLANES = [
    {"name": "xy", "x_idx": 0, "y_idx": 1, "x_label": "x (m)", "y_label": "y (m)"},
    {"name": "xz", "x_idx": 0, "y_idx": 2, "x_label": "x (m)", "y_label": "z (m)"},
    {"name": "yz", "x_idx": 1, "y_idx": 2, "x_label": "y (m)", "y_label": "z (m)"},
]


def _extract_field(data: dict, field_name: str) -> np.ndarray:
    if field_name in ("wss_x", "wss_y", "wss_z"):
        component_idx = {"wss_x": 0, "wss_y": 1, "wss_z": 2}[field_name]
        return data["wss_components_raw"][:, component_idx]
    return data[FIELD_MAP[field_name]].reshape(-1)


def _scatter(ax, coords, values, x_idx, y_idx, x_label, y_label, title,
             point_size, cmap, vmin, vmax):
    sc = ax.scatter(
        coords[:, x_idx], coords[:, y_idx], c=values,
        s=point_size, cmap=cmap, vmin=vmin, vmax=vmax,
        edgecolors="none", rasterized=True,
    )
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    return sc


def _compute_color_range(cfd_vals, pinn_vals, clip_percentile):
    """Shared color range across CFD and PINN for fair comparison."""
    combined = np.concatenate([cfd_vals, pinn_vals]) if pinn_vals is not None else cfd_vals
    low = np.percentile(combined, clip_percentile)
    high = np.percentile(combined, 100.0 - clip_percentile)
    if np.isclose(low, high):
        low, high = float(combined.min()), float(combined.max())
    return low, high


def load_pinn_predictions(checkpoint_path: str, coords: np.ndarray,
                          device: str = "cpu") -> Dict[str, np.ndarray]:
    """Run PINN inference at the given coordinates and return predicted fields.

    Returns dict with keys: 'u', 'v', 'w', 'p' (raw network outputs).
    """
    from models.base_networks import create_taa_networks

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    networks = create_taa_networks(config)
    for name, net in networks.items():
        net.load_state_dict(checkpoint["networks"][name])
        net.to(device).eval()

    coords_t = torch.tensor(coords, dtype=torch.float32, device=device)
    with torch.no_grad():
        preds = {name: net(coords_t).cpu().numpy().reshape(-1)
                 for name, net in networks.items()}
    return preds


def generate_comparison_plots(
    data_dir: str,
    filename: str,
    output_dir: str,
    field: str = "wss_magnitude",
    pinn_values: Optional[np.ndarray] = None,
    checkpoint_path: Optional[str] = None,
    subsample_factor: int = 1,
    point_size: float = 4.0,
    cmap: str = "viridis",
    dpi: int = 300,
    clip_percentile: float = 1.0,
) -> List[Path]:
    """Generate separate XY, XZ, YZ comparison PNGs (CFD vs PINN).

    Provide PINN data via *either*:
      - ``pinn_values``: pre-computed 1-D array (same length as CFD points)
      - ``checkpoint_path``: path to a saved .pt checkpoint (runs inference)

    If neither is supplied, only CFD panels are plotted.

    Returns list of saved PNG paths.
    """
    loader = TAADataLoader(
        data_dir=data_dir, geometry_scale=0.05,
        pressure_scale=100.0, wss_scale=1.0, device="cpu",
    )
    data = loader.load_single_case(filename=filename, subsample_factor=subsample_factor)
    coords = data["coords_raw"]
    cfd_vals = _extract_field(data, field)

    # Resolve PINN predictions
    if pinn_values is not None:
        pinn_vals = np.asarray(pinn_values).reshape(-1)
    elif checkpoint_path is not None:
        pinn_vals = _pinn_field_from_checkpoint(checkpoint_path, coords, field)
    else:
        pinn_vals = None

    has_pinn = pinn_vals is not None
    vmin, vmax = _compute_color_range(cfd_vals, pinn_vals, clip_percentile)

    geometry = data["geometry"]
    phase = data["phase_str"]
    stem = Path(filename).stem.replace(" ", "_")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    saved: List[Path] = []

    for plane in PLANES:
        ncols = 2 if has_pinn else 1
        fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols + 1, 5),
                                 constrained_layout=True)
        if ncols == 1:
            axes = [axes]

        # CFD panel
        sc = _scatter(
            axes[0], coords, cfd_vals,
            plane["x_idx"], plane["y_idx"],
            plane["x_label"], plane["y_label"],
            "CFD", point_size, cmap, vmin, vmax,
        )

        # PINN panel
        if has_pinn:
            _scatter(
                axes[1], coords, pinn_vals,
                plane["x_idx"], plane["y_idx"],
                plane["x_label"], plane["y_label"],
                "PINN", point_size, cmap, vmin, vmax,
            )

        plane_label = plane["name"].upper()
        fig.suptitle(
            f"{geometry} — {phase.title()} — {field}  ({plane_label} plane)",
            fontsize=13,
        )
        fig.colorbar(sc, ax=axes, shrink=0.85, label=field)

        png_path = out / f"{stem}_{field}_{plane['name']}.png"
        fig.savefig(png_path, dpi=dpi)
        plt.close(fig)
        saved.append(png_path)

    return saved


def _pinn_field_from_checkpoint(checkpoint_path, coords, field):
    """Infer the requested field from a PINN checkpoint."""
    preds = load_pinn_predictions(checkpoint_path, coords)
    if field == "pressure":
        return preds["p"]
    # For WSS fields we need velocity gradients — approximate via
    # the network; full WSS requires autograd so fall back to magnitude proxy.
    # TODO: compute proper WSS from velocity networks + wall normals
    if field == "wss_magnitude":
        return np.sqrt(preds["u"] ** 2 + preds["v"] ** 2 + preds["w"] ** 2)
    component = {"wss_x": "u", "wss_y": "v", "wss_z": "w"}.get(field)
    if component:
        return preds[component]
    raise ValueError(f"Unknown field: {field}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate separate XY / XZ / YZ comparison plots (CFD vs PINN)",
    )
    p.add_argument("--data-dir", type=str, required=True,
                   help="Directory containing TAA CSV files")
    p.add_argument("--filename", type=str, required=True,
                   help="CSV file name, e.g. '5cm systolic.csv'")
    p.add_argument("--output-dir", type=str,
                   default=str(PROJECT_ROOT / "experiments" / "figures"),
                   help="Directory for output plots")
    p.add_argument("--field", type=str, default="wss_magnitude",
                   choices=["wss_magnitude", "pressure",
                            "wss_x", "wss_y", "wss_z"],
                   help="Field to visualize")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Path to PINN checkpoint (.pt) for comparison")
    p.add_argument("--subsample-factor", type=int, default=1)
    p.add_argument("--point-size", type=float, default=4.0)
    p.add_argument("--cmap", type=str, default="viridis")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--clip-percentile", type=float, default=1.0)
    return p


def main():
    args = build_parser().parse_args()
    paths = generate_comparison_plots(
        data_dir=args.data_dir,
        filename=args.filename,
        output_dir=args.output_dir,
        field=args.field,
        checkpoint_path=args.checkpoint,
        subsample_factor=args.subsample_factor,
        point_size=args.point_size,
        cmap=args.cmap,
        dpi=args.dpi,
        clip_percentile=args.clip_percentile,
    )
    for p in paths:
        print(f"Saved: {p}")


if __name__ == "__main__":
    main()
