"""Radar geometry. x is lateral right, y forward, z up; angles are radians.

The manuscript's convolution assumes a shift-invariant sampling coordinate.
For an FFT array this is usually direction cosine sin(azimuth), not azimuth.
"""
from __future__ import annotations

import numpy as np


def cartesian_to_polar(x, y):
    x, y = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    return np.hypot(x, y), np.arctan2(x, y)


def polar_to_cartesian(distance, azimuth):
    distance, azimuth = np.broadcast_arrays(distance, azimuth)
    return distance * np.sin(azimuth), distance * np.cos(azimuth)


def points_to_indices(xyz, velocity, config):
    """Return fractional (range, velocity, angular-coordinate) indices and mask.

    ``velocity`` is a Cartesian velocity relative to the radar. Positive radial
    velocity is receding. No Doppler aliasing, elevation processing, or range
    migration is assumed. Positions outside the bin-center domain are rejected.
    """
    xyz = np.asarray(xyz, dtype=float)
    velocity = np.asarray(velocity, dtype=float)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or velocity.shape != xyz.shape:
        raise ValueError("positions and velocities must both have shape (N, 3)")
    distance = np.linalg.norm(xyz, axis=1)
    radial_velocity = np.einsum("ij,ij->i", xyz, velocity) / np.maximum(distance, 1e-12)
    theta = np.arctan2(xyz[:, 0], xyz[:, 1])
    coordinate = np.sin(theta) if config.azimuth_coordinate == "sin" else theta
    angular_grid = np.asarray(config.angular_grid)
    indices = np.column_stack((
        (distance - config.range_min) / config.range_step,
        (radial_velocity - config.velocity_min) / config.velocity_step,
        (coordinate - angular_grid[0]) / (angular_grid[1] - angular_grid[0]),
    ))
    if getattr(config, "doppler_wrap", False):
        indices[:, 1] = np.mod(indices[:, 1], config.doppler_bins)
    valid = np.isfinite(indices).all(axis=1) & (distance > 0) & (xyz[:, 1] > 0)
    upper = np.asarray(config.shape, dtype=float) - 1
    if getattr(config, "doppler_wrap", False):
        upper[1] = config.doppler_bins
    valid &= ((indices >= 0) & (indices <= upper)).all(axis=1)
    return indices, valid
