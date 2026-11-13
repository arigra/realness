"""Cartesian geometry in metres; x is lateral right, y is forward.

Oriented boxes are (center_x, center_y, width, height, yaw), with yaw radians
counterclockwise rotating the width axis from +x. For simulation.Vehicle use
theta=-vehicle.yaw. No radar-image
rectangle is treated as an equivalent ground-plane footprint.
"""
from dataclasses import asdict, dataclass
import math
import numpy as np
import torch


@dataclass(frozen=True)
class BEVGrid:
    x_min: float = -50.0
    x_max: float = 50.0
    y_min: float = 0.0
    y_max: float = 50.0
    height: int = 256
    width: int = 512

    def __post_init__(self):
        if not np.isfinite([self.x_min, self.x_max, self.y_min, self.y_max]).all():
            raise ValueError("BEV limits must be finite")
        if self.x_max <= self.x_min or self.y_max <= self.y_min:
            raise ValueError("BEV limits must increase")
        if type(self.height) is not int or type(self.width) is not int or min(self.height, self.width) < 2:
            raise ValueError("BEV dimensions must be integers >=2")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        return cls(**value)

    @property
    def x_centers(self):
        return self.x_min + (np.arange(self.width) + .5) * (self.x_max - self.x_min) / self.width

    @property
    def y_centers(self):
        return self.y_min + (np.arange(self.height) + .5) * (self.y_max - self.y_min) / self.height

    @property
    def resolution(self):
        return ((self.x_max - self.x_min) / self.width, (self.y_max - self.y_min) / self.height)

    def _constants(self, value):
        offset = [self.x_min, self.y_min, self.x_min, self.y_min]
        scale = [self.x_max - self.x_min, self.y_max - self.y_min] * 2
        if isinstance(value, torch.Tensor):
            return value.new_tensor(offset), value.new_tensor(scale)
        return np.asarray(offset), np.asarray(scale)

    def normalize_boxes(self, boxes):
        offset, scale = self._constants(boxes)
        return (boxes - offset) / scale

    def denormalize_boxes(self, boxes):
        offset, scale = self._constants(boxes)
        return boxes * scale + offset

    def coord_to_index(self, x, y):
        return ((x - self.x_min) / self.resolution[0] - .5,
                (y - self.y_min) / self.resolution[1] - .5)

    def index_to_coord(self, column, row):
        return (self.x_min + (column + .5) * self.resolution[0],
                self.y_min + (row + .5) * self.resolution[1])

    def clip_boxes(self, boxes):
        normal = self.normalize_boxes(boxes)
        normal = normal.clamp(0, 1) if isinstance(normal, torch.Tensor) else np.clip(normal, 0, 1)
        valid = (normal[..., 2:] > normal[..., :2]).all(-1)
        return self.denormalize_boxes(normal), valid


def oriented_corners(box):
    x, y, width, length, yaw = np.asarray(box, dtype=np.float64)
    if not np.isfinite([x, y, width, length, yaw]).all() or min(width, length) <= 0:
        raise ValueError("Oriented boxes require finite coordinates and positive width/length")
    local = np.array([[-width/2, -length/2], [width/2, -length/2],
                      [width/2, length/2], [-width/2, length/2]])
    c, s = math.cos(yaw), math.sin(yaw)
    return local @ np.array([[c, -s], [s, c]]).T + [x, y]


def polygon_area(vertices):
    vertices = np.asarray(vertices, dtype=np.float64).reshape(-1, 2)
    if len(vertices) < 3:
        return 0.0
    return float(abs(np.dot(vertices[:, 0], np.roll(vertices[:, 1], -1)) -
                     np.dot(vertices[:, 1], np.roll(vertices[:, 0], -1))) * .5)


def _cross(a, b):
    return float(a[0] * b[1] - a[1] * b[0])


def convex_intersection(subject, clip):
    """Sutherland-Hodgman clipping of counterclockwise convex polygons."""
    output = np.asarray(subject, dtype=np.float64).reshape(-1, 2)
    clip = np.asarray(clip, dtype=np.float64).reshape(-1, 2)
    for a, b in zip(clip, np.roll(clip, -1, axis=0)):
        vertices = output
        if not len(vertices):
            break
        output_list = []
        previous = vertices[-1]
        previous_inside = _cross(b-a, previous-a) >= -1e-10
        for current in vertices:
            inside = _cross(b-a, current-a) >= -1e-10
            if inside != previous_inside:
                direction = current - previous
                denominator = _cross(b-a, direction)
                if abs(denominator) > 1e-15:
                    t = -_cross(b-a, previous-a) / denominator
                    output_list.append(previous + np.clip(t, 0, 1) * direction)
            if inside:
                output_list.append(current)
            previous, previous_inside = current, inside
        output = np.asarray(output_list, dtype=np.float64).reshape(-1, 2)
    return output


def rotated_iou(boxes1, boxes2):
    """Exact polygon-area IoU for oriented rectangles (CPU evaluation utility)."""
    a = np.asarray(boxes1.detach().cpu() if isinstance(boxes1, torch.Tensor) else boxes1, dtype=float).reshape(-1, 5)
    b = np.asarray(boxes2.detach().cpu() if isinstance(boxes2, torch.Tensor) else boxes2, dtype=float).reshape(-1, 5)
    ac, bc = [oriented_corners(v) for v in a], [oriented_corners(v) for v in b]
    result = np.zeros((len(a), len(b)), dtype=np.float64)
    for i, p in enumerate(ac):
        for j, q in enumerate(bc):
            intersection = polygon_area(convex_intersection(p, q))
            union = a[i, 2]*a[i, 3] + b[j, 2]*b[j, 3] - intersection
            result[i, j] = np.clip(intersection / max(union, 1e-15), 0, 1)
    return result


def oriented_enclosures(boxes):
    values = np.asarray(boxes, dtype=float).reshape(-1, 5)
    return np.asarray([np.r_[p.min(0), p.max(0)] for p in map(oriented_corners, values)]).reshape(-1, 4)

