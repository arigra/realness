"""Cartesian radar detection, physical geometry and explicitly matched metrics."""
from .geometry import BEVGrid, oriented_corners, rotated_iou, oriented_enclosures
from .projection import PolarToBEV
from .models import CartesianUNet, RADDetBEV, ProbabilisticBEV, create_model
from .losses import bev_detection_loss
from .inference import detect_bev, decode_bev, rotated_nms
from .training import train_bev, load_bev_checkpoint, collate_bev
from .evaluation import evaluate_bev, evaluate_predictions_bev
from .rad_pretraining import RADDetRAD, train_rad_backbone, load_pretrained_raddet_backbone

__all__ = ["BEVGrid","PolarToBEV","CartesianUNet","RADDetBEV","ProbabilisticBEV","create_model",
           "bev_detection_loss","detect_bev","decode_bev","rotated_nms","rotated_iou",
           "oriented_corners","oriented_enclosures","train_bev","load_bev_checkpoint","collate_bev",
           "evaluate_bev","evaluate_predictions_bev","RADDetRAD","train_rad_backbone","load_pretrained_raddet_backbone"]
