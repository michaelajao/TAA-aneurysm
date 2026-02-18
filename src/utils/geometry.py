"""
Geometry utilities for TAA-PINN
Includes functions for:
- Computing wall normal vectors
- Sampling interior collocation points
- Geometric preprocessing
"""

import numpy as np
import torch
import open3d as o3d
from typing import Tuple, Optional


def compute_wall_normals(x: np.ndarray,
                         y: np.ndarray,
                         z: np.ndarray,
                         radius: float = 0.01,
                         max_nn: int = 30,
                         orient_inward: bool = True) -> np.ndarray:
    """
    Compute normal vectors for wall surface points using Open3D.

    Args:
        x: x-coordinates (N,) or (N, 1)
        y: y-coordinates (N,) or (N, 1)
        z: z-coordinates (N,) or (N, 1)
        radius: Search radius for normal estimation
        max_nn: Maximum number of nearest neighbors
        orient_inward: If True, orient normals to point inward (into fluid domain)

    Returns:
        normals: (N, 3) array of unit normal vectors
    """
    # Ensure 1D arrays
    x = x.flatten()
    y = y.flatten()
    z = z.flatten()

    # Create point cloud
    points = np.column_stack([x, y, z])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Estimate normals
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius,
            max_nn=max_nn
        )
    )

    # Extract normals
    normals = np.asarray(pcd.normals)

    # Orient normals consistently
    if orient_inward:
        # For aneurysm: normals should point inward (toward center of mass)
        centroid = points.mean(axis=0)
        vectors_to_center = centroid - points

        # Check if normals point toward center; if not, flip them
        dot_products = (normals * vectors_to_center).sum(axis=1)
        flip_mask = dot_products < 0
        normals[flip_mask] = -normals[flip_mask]

    # Validate: normals should be unit vectors
    norms = np.linalg.norm(normals, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3), f"Normals not unit vectors: {norms.min():.4f} to {norms.max():.4f}"

    return normals


def compute_wall_normals_torch(x: torch.Tensor,
                               y: torch.Tensor,
                               z: torch.Tensor,
                               radius: float = 0.01,
                               max_nn: int = 30,
                               orient_inward: bool = True,
                               device: str = 'cuda') -> torch.Tensor:
    """
    Compute wall normals and return as PyTorch tensor.

    Args:
        x, y, z: PyTorch tensors of coordinates
        radius: Search radius for normal estimation
        max_nn: Maximum number of nearest neighbors
        orient_inward: If True, orient normals inward
        device: Device for output tensor

    Returns:
        normals: (N, 3) PyTorch tensor of unit normal vectors
    """
    # Convert to numpy
    x_np = x.detach().cpu().numpy()
    y_np = y.detach().cpu().numpy()
    z_np = z.detach().cpu().numpy()

    # Compute normals
    normals_np = compute_wall_normals(x_np, y_np, z_np, radius, max_nn, orient_inward)

    # Convert back to torch
    normals = torch.tensor(normals_np, dtype=torch.float32, device=device)

    return normals


def sample_interior_points(x_wall: np.ndarray,
                           y_wall: np.ndarray,
                           z_wall: np.ndarray,
                           n_samples: int = 10000,
                           offset_range: Tuple[float, float] = (0.05, 0.5),
                           normals: Optional[np.ndarray] = None,
                           seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sample interior collocation points for physics loss.

    Strategy: Move inward from wall points along normal directions.

    Args:
        x_wall, y_wall, z_wall: Wall coordinates
        n_samples: Number of interior points to sample
        offset_range: (min, max) offset distance from wall (in normalized coordinates)
        normals: Pre-computed normal vectors (if None, will compute)
        seed: Random seed for reproducibility

    Returns:
        x_interior, y_interior, z_interior: Interior point coordinates
    """
    if seed is not None:
        np.random.seed(seed)

    # Flatten arrays
    x_wall = x_wall.flatten()
    y_wall = y_wall.flatten()
    z_wall = z_wall.flatten()

    n_wall_points = len(x_wall)

    # Compute normals if not provided
    if normals is None:
        normals = compute_wall_normals(x_wall, y_wall, z_wall)

    # Randomly select wall points
    indices = np.random.choice(n_wall_points, size=n_samples, replace=True)

    # Random offsets along normal direction
    offsets = np.random.uniform(offset_range[0], offset_range[1], size=n_samples)

    # Compute interior points
    wall_points = np.column_stack([x_wall[indices], y_wall[indices], z_wall[indices]])
    normal_vectors = normals[indices]

    interior_points = wall_points + offsets[:, None] * normal_vectors

    x_interior = interior_points[:, 0]
    y_interior = interior_points[:, 1]
    z_interior = interior_points[:, 2]

    return x_interior, y_interior, z_interior


def sample_interior_points_torch(x_wall: torch.Tensor,
                                 y_wall: torch.Tensor,
                                 z_wall: torch.Tensor,
                                 n_samples: int = 10000,
                                 offset_range: Tuple[float, float] = (0.05, 0.5),
                                 normals: Optional[torch.Tensor] = None,
                                 seed: Optional[int] = None,
                                 device: str = 'cuda') -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sample interior points and return as PyTorch tensors.

    Args:
        x_wall, y_wall, z_wall: Wall coordinates as tensors
        n_samples: Number of samples
        offset_range: Offset range from wall
        normals: Pre-computed normals as tensor
        seed: Random seed
        device: Device for output

    Returns:
        x_interior, y_interior, z_interior: Interior coordinates as tensors
    """
    # Convert to numpy
    x_np = x_wall.detach().cpu().numpy()
    y_np = y_wall.detach().cpu().numpy()
    z_np = z_wall.detach().cpu().numpy()

    normals_np = None
    if normals is not None:
        normals_np = normals.detach().cpu().numpy()

    # Sample interior points
    x_int, y_int, z_int = sample_interior_points(
        x_np, y_np, z_np,
        n_samples=n_samples,
        offset_range=offset_range,
        normals=normals_np,
        seed=seed
    )

    # Convert to torch
    x_interior = torch.tensor(x_int.reshape(-1, 1), dtype=torch.float32, device=device)
    y_interior = torch.tensor(y_int.reshape(-1, 1), dtype=torch.float32, device=device)
    z_interior = torch.tensor(z_int.reshape(-1, 1), dtype=torch.float32, device=device)

    return x_interior, y_interior, z_interior
