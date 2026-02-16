"""
TAA Data Loader for CSV Files
Parses TAA CFD data from CSV format and prepares tensors for PINN training
"""

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
    - Coordinate normalization and centering
    - Extraction of wall boundary data (coordinates, pressure, WSS)
    - Preparation of PyTorch tensors
    """

    def __init__(self,
                 data_dir: str,
                 geometry_scale: float = 0.05,  # 5cm default
                 pressure_scale: float = 100.0,  # Pa
                 wss_scale: float = 1.0,  # Pa
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the TAA data loader.

        Args:
            data_dir: Directory containing CSV files
            geometry_scale: Characteristic length for normalization (m)
            pressure_scale: Pressure scale for normalization (Pa)
            wss_scale: WSS scale for normalization (Pa)
            device: Device for tensor placement ('cuda' or 'cpu')
        """
        self.data_dir = data_dir
        self.geometry_scale = geometry_scale
        self.pressure_scale = pressure_scale
        self.wss_scale = wss_scale
        self.device = device

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

    def load_single_case(self,
                        filename: str,
                        subsample_factor: int = 1) -> Dict[str, np.ndarray]:
        """
        Load a single CSV file and extract relevant data.

        Coordinate centering uses the FULL dataset mean (before subsampling)
        so that the centering is consistent regardless of subsample_factor.

        Args:
            filename: Name of CSV file (e.g., '5cm diastolic.csv')
            subsample_factor: Sample every Nth point (1 = all points)

        Returns:
            Dictionary containing:
                - coords: (N, 3) array of x, y, z coordinates
                - pressure: (N, 1) array of pressure values
                - wss_magnitude: (N, 1) array of WSS magnitude
                - wss_components: (N, 3) array of WSS x, y, z components
                - phase: Scalar (0.0 for diastolic, 1.0 for systolic)
                - geometry: String geometry code
                - coords_mean: (3,) the centering mean (from full data)
        """
        filepath = os.path.join(self.data_dir, filename)

        # Read CSV with custom header (skip [Name] and [Data] rows)
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # Find the header row (contains column names)
        header_idx = None
        for i, line in enumerate(lines):
            if 'X [ m ]' in line:
                header_idx = i
                break

        if header_idx is None:
            raise ValueError(f"Could not find header in {filename}")

        # Read data starting AFTER header row (header_idx is the header, data starts at header_idx+1)
        # But pandas will use header_idx row as column names if we use header=0
        df = pd.read_csv(filepath, skiprows=list(range(header_idx)), low_memory=False)

        # Extract columns (handle potential whitespace in column names and values)
        df.columns = df.columns.str.strip()

        # Strip whitespace from all string values
        df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

        # Convert all numeric columns to float
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Drop rows with any NaN values (these are likely header rows or empty rows)
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

        # Center coordinates using full-dataset mean (consistent across subsample factors)
        coords_centered = coords - coords_mean

        # Normalize by geometry scale
        coords_normalized = coords_centered / self.geometry_scale

        # Extract pressure
        pressure = df['Pressure [ Pa ]'].values.astype(float).reshape(-1, 1)
        pressure_normalized = pressure / self.pressure_scale

        # Extract WSS components
        wss_x = df['Wall Shear X [ Pa ]'].values.astype(float)
        wss_y = df['Wall Shear Y [ Pa ]'].values.astype(float)
        wss_z = df['Wall Shear Z [ Pa ]'].values.astype(float)
        wss_components = np.column_stack([wss_x, wss_y, wss_z])
        wss_components_normalized = wss_components / self.wss_scale

        # Calculate WSS magnitude
        wss_magnitude = df['Wall Shear [ Pa ]'].values.astype(float).reshape(-1, 1)
        wss_magnitude_normalized = wss_magnitude / self.wss_scale

        # Determine phase from filename
        phase_str = 'systolic' if 'systolic' in filename.lower() else 'diastolic'
        phase_value = self.phase_map[phase_str]

        # Determine geometry from filename
        geometry_code = self._parse_geometry_from_filename(filename)

        return {
            'coords': coords_normalized,
            'coords_raw': coords,  # Keep raw for visualization
            'coords_mean': coords_mean,  # Full-data centering mean
            'pressure': pressure_normalized,
            'pressure_raw': pressure,
            'wss_magnitude': wss_magnitude_normalized,
            'wss_magnitude_raw': wss_magnitude,
            'wss_components': wss_components_normalized,
            'wss_components_raw': wss_components,
            'phase': phase_value,
            'phase_str': phase_str,
            'geometry': geometry_code,
            'n_points': coords_normalized.shape[0]
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
        # List all CSV files in directory
        csv_files = [f for f in os.listdir(self.data_dir) if f.endswith('.csv')]

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

        # Coordinates
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

        # Pressure
        pressure = torch.tensor(data['pressure'], dtype=torch.float32, device=self.device)
        tensors['pressure'] = pressure

        # WSS components
        wss_x = torch.tensor(data['wss_components'][:, 0:1], dtype=torch.float32, device=self.device)
        wss_y = torch.tensor(data['wss_components'][:, 1:2], dtype=torch.float32, device=self.device)
        wss_z = torch.tensor(data['wss_components'][:, 2:3], dtype=torch.float32, device=self.device)

        tensors['wss_x'] = wss_x
        tensors['wss_y'] = wss_y
        tensors['wss_z'] = wss_z

        # WSS magnitude
        wss_mag = torch.tensor(data['wss_magnitude'], dtype=torch.float32, device=self.device)
        tensors['wss_magnitude'] = wss_mag

        return tensors

    def get_statistics(self, data: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Compute statistics for a loaded dataset.

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
