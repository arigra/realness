"""Differentiable coordinate-aware polar intensity -> Cartesian sampling.

This interpolates an intensity image at BEV pixel centers. It is not an
area-conserving redistribution of RCS and does not recover missing resolution.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .geometry import BEVGrid


class PolarToBEV(nn.Module):
    def __init__(self, ranges, angular_values, grid: BEVGrid, azimuth_coordinate="sin"):
        super().__init__()
        ranges, angular = np.asarray(ranges, dtype=float), np.asarray(angular_values, dtype=float)
        if ranges.ndim != 1 or angular.ndim != 1 or min(len(ranges), len(angular)) < 2:
            raise ValueError("Need one-dimensional range and angular axes with >=2 bins")
        if not np.isfinite(ranges).all() or not np.isfinite(angular).all() or np.any(np.diff(ranges) <= 0) or np.any(np.diff(angular) <= 0):
            raise ValueError("Projection axes must be finite and strictly increasing")
        if azimuth_coordinate not in ("sin", "angle"):
            raise ValueError("azimuth_coordinate must be sin or angle")
        if ranges[0] < 0 or (azimuth_coordinate == "sin" and np.any(np.abs(angular) > 1)):
            raise ValueError("Invalid range or direction-cosine coordinates")
        xx, yy = np.meshgrid(grid.x_centers, grid.y_centers)
        rr, theta = np.hypot(xx, yy), np.arctan2(xx, yy)
        coordinate = np.sin(theta) if azimuth_coordinate == "sin" else theta
        # sin(theta) alone is ambiguous behind the radar; front hemisphere only.
        valid = (yy >= 0) & (rr >= ranges[0]) & (rr <= ranges[-1]) & (coordinate >= angular[0]) & (coordinate <= angular[-1])
        ri = np.interp(rr.ravel(), ranges, np.arange(len(ranges))).reshape(rr.shape)
        ai = np.interp(coordinate.ravel(), angular, np.arange(len(angular))).reshape(rr.shape)
        sampling = np.stack([2*ai/(len(angular)-1)-1, 2*ri/(len(ranges)-1)-1], axis=-1)
        self.register_buffer("sampling_grid", torch.tensor(sampling, dtype=torch.float32).unsqueeze(0))
        self.register_buffer("valid_mask", torch.tensor(valid, dtype=torch.bool))
        self.input_shape = (len(ranges), len(angular))
        self.grid = grid

    def forward(self, polar):
        polar = torch.as_tensor(polar)
        if polar.is_complex():
            raise ValueError("Project nonnegative linear intensity/power, not complex fields")
        if not polar.is_floating_point():
            polar = polar.float()
        if not torch.isfinite(polar).all():
            raise ValueError("Projection input must be finite")
        if tuple(polar.shape[-2:]) != self.input_shape or polar.ndim not in (2, 3, 4):
            raise ValueError("Input must be RA, BRA, or BCRA on the configured axes")
        rank = polar.ndim
        source = polar[None, None] if rank == 2 else polar[:, None] if rank == 3 else polar
        sample = self.sampling_grid.to(device=source.device, dtype=source.dtype).expand(source.shape[0], -1, -1, -1)
        output = F.grid_sample(source, sample, mode="bilinear", padding_mode="zeros", align_corners=True)
        output = output * self.valid_mask.to(device=source.device, dtype=source.dtype)[None, None]
        return output[0, 0] if rank == 2 else output[:, 0] if rank == 3 else output
