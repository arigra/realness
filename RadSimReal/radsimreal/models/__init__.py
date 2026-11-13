"""Documented reconstruction of the coordinate-augmented U-Net detector family.

The unpublished RadSimReal detection heads and training recipe are not recoverable
from the publication. These modules implement an explicit research baseline;
they are not an authors' model or a port of RADDet/Probabilistic.
"""

from .unet import CoordinateUNet
from .losses import build_targets, detection_loss
from .inference import detect, decode_detections

__all__ = ["CoordinateUNet", "build_targets", "detection_loss", "detect", "decode_detections"]
