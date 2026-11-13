import torch

from radsimreal.bev.geometry import BEVGrid
from radsimreal.bev.inference import decode_presence
from radsimreal.bev.losses import presence_loss, presence_targets
from radsimreal.bev.models import create_model


def test_presence_targets_decode_back_to_the_boxes():
    grid = BEVGrid()
    physical = torch.tensor([[-10.0, 20.0, -6.0, 24.5], [5.0, 40.0, 7.0, 41.0], [30.0, 10.0, 30.1, 10.1]])
    boxes = [torch.as_tensor(grid.normalize_boxes(physical.numpy()))]
    outputs = {"presence": torch.zeros(1, 1, grid.height, grid.width), "box": torch.zeros(1, 4, grid.height, grid.width),
               "box_scale": 32.0, "head_type": "presence"}
    mask, target = presence_targets(outputs, boxes)
    assert mask.sum() > 0
    assert mask[0].bool().sum() >= 3  # the tiny third box still owns a pixel
    outputs["presence"] = (mask[:, None] * 20 - 10)
    outputs["box"] = target
    decoded = decode_presence(outputs, grid, score_threshold=.5, nms_threshold=.3, top_k=10)[0]
    assert len(decoded["boxes"]) == 3
    for b in physical:
        assert (decoded["boxes"].double() - b.double()).abs().sum(-1).min() < 1e-3


def test_zhang_unet_trains_one_step():
    grid = BEVGrid(height=32, width=64)
    model = create_model("unet_zhang", grid=grid, base_channels=4)
    image = torch.rand(2, 3, 32, 64)
    boxes = [torch.tensor([[.1, .2, .3, .4]]), torch.zeros(0, 4)]
    loss = presence_loss(model(image), boxes)
    loss["total"].backward()
    assert torch.isfinite(loss["total"]) and set(loss) >= {"total", "heatmap", "box", "bce", "dice"}
