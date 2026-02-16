"""
TAA Data Loader for CSV Files

Parses TAA CFD data from CSV format, computes non-dimensional reference scales
from the data and fluid properties, and prepares tensors for PINN training.

Non-dimensionalization:
    x_bar = (x - x_mean) / L_ref
    u_bar = u / U_ref
    p_bar = p / P_ref        where P_ref = rho * U_ref^2
    tau_bar = tau / tau_ref   where tau_ref = mu * U_ref / L_ref

    U_ref = sqrt(max|p| / rho)   (characteristic velocity from pressure data)
    Re = rho * U_ref * L_ref / mu (Reynolds number)
"""

import math
import numpy as np
import pandas as pd
import torch
import os
from typing import Dict, Tuple, Optional, List


class TAADataLoader:
    """
    Data loader for Thoracic Aortic Aneurysm CFD data in CSV format.

    Handles:
    - Parsing CSV files with custom header format
    - Computing non-dimensional reference scales from data and fluid properties
    - Coordinate normalization and centering
    - Extraction of wall boundary data (coordinates, pressure, WSS)
    - Preparation of PyTorch tensors
    """

    def __init__(self,
                 data_dir: str,
                 L_ref: float = 0.05,
                 rho: float = 1060.0,
                 mu: float = 0.0035,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the TAA data loader.

        Args:
            data_dir: Directory containing CSV files
            L_ref: Characteristic length for normalization (m), e.g. vessel diameter
            rho: Fluid density (kg/m^3)
            mu: Dynamic viscosity (Pa.s)
            device: Device for tensor placement ('cuda' or 'cpu')
        """
        self.data_dir = data_dir
        self.L_ref = L_ref
        self.rho = rho
        self.mu = mu
        self.device = device

        # Reference scales (populated by compute_reference_scales)
        self.U_ref = None
        self.P_ref = None
        self.tau_ref = None
        self.Re = None

        # Geometry code mapping
        self.geometry_map = {
            '5cm': 'AS5',
            '5cm ASU': 'PD5',
            '5cm ASD': 'AD5',
            '6cm': 'AS6',
            '6cm ASU': 'PD6',
            '6cm ASD': 'AD6'
        }

        # Phase encoding
        self.phase_map = {
            'systolic': 1.0,
            'diastolic': 0.0
        }

    def compute_reference_scales(self, filenames: List[str]) -> Dict[str, float]:
        """
        Compute non-dimensional reference scales from the training data.

        First pass over all specified files to find max(|pressure|), then derive:
            U_ref = sqrt(max|p| / rho)
            P_ref = rho * U_ref^2   (= max|p| by construction)
            tau_ref = mu * U_ref / L_ref
            Re = rho * U_ref * L_ref / mu

        Args:
            filenames: List of CSV filenames to scan for pressure range

        Returns:
            Dictionary with U_ref, P_ref, tau_ref, Re
        """
        max_abs_pressure = 0.0

        for filename in filenames:
            filepath = os.path.join(self.data_dir, filename)
            header_idx = self._find_header_row(filepath)
            df = pd.read_csv(filepath, skiprows=list(range(header_idx)), low_memory=False)
            df.columns = df.columns.str.strip()
            df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df = df.dropna().reset_index(drop=True)

            pressure = df['Pressure [ Pa ]'].values.astype(float)
            file_max = np.abs(pressure).max()
            max_abs_pressure = max(max_abs_pressure, file_max)

        # Derive reference scales
        self.U_ref = math.sqrt(max_abs_pressure / self.rho)
        self.P_ref = self.rho * self.U_ref ** 2  # = max_abs_pressure
        self.tau_ref = self.mu * self.U_ref / self.L_ref
        self.Re = self.rho * self.U_ref * self.L_ref / self.mu

        scales = {
            'L_ref': self.L_ref,
            'U_ref': self.U_ref,
            'P_ref': self.P_ref,
            'tau_ref': self.tau_ref,
            'Re': self.Re,
            'rho': self.rho,
            'mu': self.mu,
            'max_abs_pressure_Pa': max_abs_pressure,
        }

        print("\n  Non-dimensional reference scales:")
        print(f"    L_ref     = {self.L_ref:.4f} m")
        print(f"    U_ref     = {self.U_ref:.4f} m/s  (from max|p|={max_abs_pressure:.2f} Pa)")
        print(f"    P_ref     = {self.P_ref:.2f} Pa  (= rho * U_ref^2)")
        print(f"    tau_ref   = {self.tau_ref:.6f} Pa  (= mu * U_ref / L_ref)")
        print(f"    Re        = {self.Re:.1f}")
        print(f"    rho       = {self.rho:.1f} kg/m^3")
        print(f"    mu        = {self.mu:.6f} Pa.s")

        return scales

    def _find_header_row(self, filepath: str) -> int:
        """Find the row index containing column headers in a CSV file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if 'X [ m ]' in line:
                    return i
        raise ValueError(f"Could not find header in {filepath}")

    def load_single_case(self,
                        filename: str,
                        subsample_factor: int = 1) -> Dict[str, np.ndarray]:
        """
        Load a single CSV file and extract relevant data.

        Coordinate centering uses the FULL dataset mean (before subsampling)
        so that the centering is consistent regardless of subsample_factor.

        Requires compute_reference_scales() to have been called first.

        Args:
            filename: Name of CSV file (e.g., '5cm diastolic.csv')
            subsample_factor: Sample every Nth point (1 = all points)

        Returns:
            Dictionary containing normalized (non-dimensional) data and raw values.
        """
        if self.U_ref is None:
            raise RuntimeError(
                "Reference scales not computed. Call compute_reference_scales() first."
            )

        filepath = os.path.join(self.data_dir, filename)

        # Read CSV with custom header
        header_idx = self._find_header_row(filepath)
        df = pd.read_csv(filepath, skiprows=list(range(header_idx)), low_memory=False)

        # Clean columns
        df.columns = df.columns.str.strip()
        df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna().reset_index(drop=True)

        # Compute centering mean from FULL dataset BEFORE subsampling
        x_full = df['X [ m ]'].values.astype(float)
        y_full = df['Y [ m ]'].values.astype(float)
        z_full = df['Z [ m ]'].values.astype(float)
        coords_full = np.column_stack([x_full, y_full, z_full])
        coords_mean = coords_full.mean(axis=0)

        # Subsample dataframe if requested (AFTER computing the full mean)
        if subsample_factor > 1:
            df = df.iloc[::subsample_factor].reset_index(drop=True)

        # Extract coordinates
        x = df['X [ m ]'].values.astype(float)
        y = df['Y [ m ]'].values.astype(float)
        z = df['Z [ m ]'].values.astype(float)
        coords = np.column_stack([x, y, z])

        # Non-dimensionalize coordinates: x_bar = (x - x_mean) / L_ref
        coords_centered = coords - coords_mean
        coords_nondim = coords_centered / self.L_ref

        # Non-dimensionalize pressure: p_bar = p / P_ref
        pressure = df['Pressure [ Pa ]'].values.astype(float).reshape(-1, 1)
        pressure_nondim = pressure / self.P_ref

        # Non-dimensionalize WSS: tau_bar = tau / tau_ref
        wss_x = df['Wall Shear X [ Pa ]'].values.astype(float)
        wss_y = df['Wall Shear Y [ Pa ]'].values.astype(float)
        wss_z = df['Wall Shear Z [ Pa ]'].values.astype(float)
        wss_components = np.column_stack([wss_x, wss_y, wss_z])
        wss_components_nondim = wss_components / self.tau_ref

        wss_magnitude = df['Wall Shear [ Pa ]'].values.astype(float).reshape(-1, 1)
        wss_magnitude_nondim = wss_magnitude / self.tau_ref

        # Determine phase from filename
        phase_str = 'systolic' if 'systolic' in filename.lower() else 'diastolic'
        phase_value = self.phase_map[phase_str]

        # Determine geometry from filename
        geometry_code = self._parse_geometry_from_filename(filename)

        return {
            'coords': coords_nondim,
            'coords_raw': coords,
            'coords_mean': coords_mean,
            'pressure': pressure_nondim,
            'pressure_raw': pressure,
            'wss_magnitude': wss_magnitude_nondim,
            'wss_magnitude_raw': wss_magnitude,
            'wss_components': wss_components_nondim,
            'wss_components_raw': wss_components,
            'phase': phase_value,
            'phase_str': phase_str,
            'geometry': geometry_code,
            'n_points': coords_nondim.shape[0]
        }

    def _parse_geometry_from_filename(self, filename: str) -> str:
        """Parse geometry code from filename."""
        filename_lower = filename.lower()

        if '5cm asu' in filename_lower:
            return 'PD5'
        elif '5cm asd' in filename_lower:
            return 'AD5'
        elif '5cm' in filename_lower:
            return 'AS5'
        elif '6cm asu' in filename_lower:
            return 'PD6'
        elif '6cm asd' in filename_lower:
            return 'AD6'
        elif '6cm' in filename_lower:
            return 'AS6'
        else:
            raise ValueError(f"Cannot parse geometry from filename: {filename}")

    def load_all_geometries(self,
                           subsample_factor: int = 1) -> Dict[Tuple[str, str], Dict]:
        """
        Load all 12 geometry-phase combinations.

        Args:
            subsample_factor: Sample every Nth point

        Returns:
            Dictionary with keys (geometry_code, phase) and values from load_single_case()
        """
        csv_files = [f for f in os.listdir(self.data_dir) if f.endswith('.csv')]

        # Compute reference scales from all files
        self.compute_reference_scales(csv_files)

        all_data = {}
        for filename in csv_files:
            try:
                data = self.load_single_case(filename, subsample_factor)
                key = (data['geometry'], data['phase_str'])
                all_data[key] = data
                print(f"Loaded {filename}: {data['n_points']} points")
            except Exception as e:
                print(f"Error loading {filename}: {e}")

        return all_data

    def prepare_tensors(self,
                       data: Dict[str, np.ndarray],
                       include_phase: bool = True) -> Dict[str, torch.Tensor]:
        """
        Convert numpy arrays to PyTorch tensors on specified device.

        Args:
            data: Output from load_single_case()
            include_phase: Whether to include phase in input features

        Returns:
            Dictionary of PyTorch tensors
        """
        tensors = {}

        # Coordinates (non-dimensional)
        x = torch.tensor(data['coords'][:, 0:1], dtype=torch.float32, device=self.device)
        y = torch.tensor(data['coords'][:, 1:2], dtype=torch.float32, device=self.device)
        z = torch.tensor(data['coords'][:, 2:3], dtype=torch.float32, device=self.device)

        tensors['x'] = x
        tensors['y'] = y
        tensors['z'] = z

        # Phase (repeated for all points)
        if include_phase:
            phase = torch.full((x.shape[0], 1), data['phase'],
                             dtype=torch.float32, device=self.device)
            tensors['phase'] = phase

        # Pressure (non-dimensional)
        pressure = torch.tensor(data['pressure'], dtype=torch.float32, device=self.device)
        tensors['pressure'] = pressure

        # WSS components (non-dimensional, viscous scaling)
        wss_x = torch.tensor(data['wss_components'][:, 0:1], dtype=torch.float32, device=self.device)
        wss_y = torch.tensor(data['wss_components'][:, 1:2], dtype=torch.float32, device=self.device)
        wss_z = torch.tensor(data['wss_components'][:, 2:3], dtype=torch.float32, device=self.device)

        tensors['wss_x'] = wss_x
        tensors['wss_y'] = wss_y
        tensors['wss_z'] = wss_z

        # WSS magnitude (non-dimensional)
        wss_mag = torch.tensor(data['wss_magnitude'], dtype=torch.float32, device=self.device)
        tensors['wss_magnitude'] = wss_mag

        return tensors

    def get_statistics(self, data: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Compute statistics for a loaded dataset (non-dimensional values).

        Args:
            data: Output from load_single_case()

        Returns:
            Dictionary of statistics
        """
        stats = {
            'n_points': data['n_points'],
            'coord_min': data['coords'].min(axis=0),
            'coord_max': data['coords'].max(axis=0),
            'coord_mean': data['coords'].mean(axis=0),
            'coord_std': data['coords'].std(axis=0),
            'pressure_min': data['pressure'].min(),
            'pressure_max': data['pressure'].max(),
            'pressure_mean': data['pressure'].mean(),
            'wss_min': data['wss_magnitude'].min(),
            'wss_max': data['wss_magnitude'].max(),
            'wss_mean': data['wss_magnitude'].mean(),
            'wss_std': data['wss_magnitude'].std(),
        }

        return stats
