"""Cartesian detection models with declared primary-source implementation limits.

RADDet architecture follows the released RadarResNet and Cartesian YOLO head;
the upstream README warns that its current backbone can differ from the paper.
Dong's ResNet18 encoder, skip decoder, three anchors and heteroscedastic head
follow arXiv:2004.05310 Sections 4.1-4.4. Unknown channel/assignment choices and
RadSimReal adaptations are disclosed in docs/DETECTOR_IMPLEMENTATION.md.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from ..models.unet import CoordinateUNet
from .geometry import BEVGrid


class SafeBatchNorm2d(nn.BatchNorm2d):
    """Use stored moments only for degenerate one-value/channel pilot batches."""
    def forward(self, x):
        if self.training and x.numel() // x.shape[1] <= 1:
            return F.batch_norm(x, self.running_mean, self.running_var, self.weight,
                                self.bias, False, self.momentum, self.eps)
        return super().forward(x)


def conv_bn(in_ch, out_ch, kernel=3, stride=1, keras=False):
    return nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel, stride, kernel//2, bias=keras),
                         SafeBatchNorm2d(out_ch, eps=.001 if keras else 1e-5,
                                         momentum=.01 if keras else .1), nn.ReLU(inplace=True))


class CartesianUNet(CoordinateUNet):
    architecture = "coordinate_unet_bev"
    requires_radar = False
    head_type = "center"
    oriented = False
    fidelity = "U-Net family; explicit reconstructed RadSimReal Cartesian adaptation"

    def __init__(self, grid=None, base_channels=16, **kwargs):
        super().__init__(base_channels=base_channels, in_channels=3)
        self.grid = grid or BEVGrid()
        self.export_config = {"name": "unet", "grid": self.grid.to_dict(), "base_channels": base_channels}

    def forward(self, image, radar=None):
        output = super().forward(image)
        output["head_type"] = "center"
        return output


class ZhangUNet(CoordinateUNet):
    """U-Net with Zhang et al. (ICASSP 2020, arXiv:1902.05394) Sec. 4.3 outputs, adapted to boxes.

    Zhang: an object-presence map Cp (per-cell occupancy probability, trained as segmentation with
    BCE + Dice) and per-cell regression maps (Cx, Cy) trained with MSE; the three losses summed.
    RadSimReal Sec. 4 cites this U-Net with an added Cartesian-coordinate input, and its Fig. 5 shows
    filled box-shaped score maps. Adaptation (ours, unstated in both papers): presence target = pixels
    inside a ground-truth box; the regression maps carry that box's centre offset and size, in pixels/32.
    """
    architecture = "zhang_presence_unet_bev"
    requires_radar = False
    head_type = "presence"
    oriented = False
    fidelity = "Zhang2020 presence+regression U-Net; box adaptation is ours"
    box_scale = 32.0

    def __init__(self, grid=None, base_channels=16, **kwargs):
        super().__init__(base_channels=base_channels, in_channels=3)
        self.grid = grid or BEVGrid()
        del self.heatmap_head, self.offset_head, self.size_head
        self.presence_head = nn.Conv2d(base_channels, 1, 1)
        self.box_head = nn.Conv2d(base_channels, 4, 1)
        self.export_config = {"name": "unet_zhang", "grid": self.grid.to_dict(), "base_channels": base_channels}

    def forward(self, image, radar=None):
        if image.ndim != 4 or image.shape[1] != self.in_channels:
            raise ValueError(f"expected Bx{self.in_channels}xHxW input")
        x = self.stem(image)
        skips = [x]
        for down in self.down:
            x = down(self.pool(x))
            skips.append(x)
        for up, skip in zip(self.up, reversed(skips[:-1])):
            x = up(x, skip)
        x = self.refine(x)
        return {"head_type": self.head_type, "presence": self.presence_head(x), "box": self.box_head(x),
                "box_scale": self.box_scale}


class RadarResidual(nn.Module):
    """Released RADDet basicResidualBlock: 3x3,3x3,1x1 and optional3x3 skip."""
    def __init__(self, channels, expansion=1):
        super().__init__()
        out = int(channels * expansion)
        self.path = nn.Sequential(conv_bn(channels, channels, keras=True),
                                  conv_bn(channels, out, keras=True),
                                  conv_bn(out, out, kernel=1, keras=True))
        self.skip = conv_bn(channels, out, keras=True) if out != channels else nn.Identity()

    def forward(self, x):
        return self.path(x) + self.skip(x)


class RADDetBEV(nn.Module):
    """PyTorch port of released RADDet backbone + Cartesian 2D YOLO head.

    radar input B,D,R,A is already real log-normalized power, matching the
    released Doppler-as-channel convolution. The learned spatial MLP converts
    R/16 x A/16 maps to R/16 x 2*A/16. The public source's original Cartesian
    output has forward then lateral axes; here they are explicitly mapped to
    normalized lateral-x / forward-y boxes by decoding (not by relabeling GT).
    """
    architecture = "raddet_cartesian_port"
    requires_radar = True
    head_type = "raddet_anchor"
    oriented = False
    fidelity = "released RADDet RadarResNet+Cartesian head port; class-car/grid adaptation; no pretrained parity claim"

    def __init__(self, grid=None, input_shape=(64, 256, 256), anchors=None, repeats=(2,4,8,16), upstream_size_clamp=False, **kwargs):
        super().__init__()
        self.grid = grid or BEVGrid()
        self.upstream_size_clamp = bool(upstream_size_clamp)
        self.input_shape = tuple(int(n) for n in input_shape)
        if len(self.input_shape) != 3 or min(self.input_shape) < 1 or min(self.input_shape[1:]) < 16:
            raise ValueError("RADDet input_shape must be (D,R,A), with R,A>=16")
        if len(repeats) != 4 or any(type(n) is not int or n < 1 for n in repeats):
            raise ValueError("RADDet requires four positive residual repeat counts")
        channels = self.input_shape[0]
        stages = []
        for stage, count in enumerate(repeats):
            blocks = []
            for repeat in range(count):
                expansion = 2 if stage >= 2 and repeat == count-1 else 1
                blocks.append(RadarResidual(channels, expansion))
                channels *= expansion
            blocks.append(nn.MaxPool2d(2, 2, ceil_mode=True))
            stages.append(nn.Sequential(*blocks))
        self.backbone = nn.Sequential(*stages)
        fr, fa = math.ceil(self.input_shape[1]/16), math.ceil(self.input_shape[2]/16)
        self.feature_shape = (fr, fa)
        self.spatial_transform = nn.Sequential(nn.Linear(fr*fa, 2*fr*fa), nn.ReLU(),
                                              nn.Linear(2*fr*fa, 2*fr*fa), nn.ReLU())
        self.cart_residual = RadarResidual(channels)
        self.neck = conv_bn(channels, channels*2, keras=True)
        # Released cart anchors are (forward pixels,lateral pixels) on (256,512).
        # Reorder to our lateral/forward normalized convention, preserving source.
        source_anchors = [(11,15), (28,24), (6,8), (43,41), (17,27), (21,18)]
        if anchors is None:
            anchors = [(lateral/512, forward/256) for forward, lateral in source_anchors]
        anchors = torch.as_tensor(anchors, dtype=torch.float32)
        if anchors.ndim != 2 or anchors.shape[1] != 2 or not torch.isfinite(anchors).all() or (anchors <= 0).any():
            raise ValueError("anchors must be positive normalized Cartesian width/height pairs")
        self.register_buffer("anchors", anchors)
        self.head = nn.Conv2d(channels*2, len(anchors)*6, 1)
        self.export_config = {"name": "raddet", "grid": self.grid.to_dict(), "input_shape": list(self.input_shape),
                              "anchors": anchors.tolist(), "repeats": list(repeats),
                              "upstream_size_clamp": self.upstream_size_clamp}
        self.apply(_xavier)

    def forward(self, image, radar=None):
        if radar is None or radar.ndim != 4 or tuple(radar.shape[1:]) != self.input_shape:
            raise ValueError(f"RADDet requires normalized radar B,D,R,A with shape {self.input_shape}")
        features = self.backbone(radar)
        b, c, h, w = features.shape
        cart = self.spatial_transform(features.flatten(2)).reshape(b, c, h, 2*w)
        cart = self.neck(self.cart_residual(cart))
        raw = self.head(cart).reshape(b, len(self.anchors), 6, h, 2*w).permute(0,3,4,1,2)
        return {"head_type": self.head_type, "raw": raw, "anchors": self.anchors,
                "upstream_size_clamp": self.upstream_size_clamp,
                "cartesian_axes": "row=forward,column=lateral"}


def _xavier(module):
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class BasicResNetBlock(nn.Module):
    """ResNet18 two-convolution unit; paper calls these bottleneck blocks."""
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.first = conv_bn(in_ch, out_ch, stride=stride)
        self.second = nn.Sequential(nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False), SafeBatchNorm2d(out_ch))
        self.skip = nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, stride, bias=False), SafeBatchNorm2d(out_ch)) if stride != 1 or in_ch != out_ch else nn.Identity()

    def forward(self, x):
        return F.relu(self.second(self.first(x)) + self.skip(x), inplace=True)


class ProbabilisticBEV(nn.Module):
    """Dong2020 encoder/skip decoder and 3 oriented uncertainty-bearing anchors.

    Per anchor: object logit, six offsets (x,y,logw,logh,cosd,sind), six log
    variance outputs. Without actual yaw labels, oriented=False explicitly
    disables angle supervision/evaluation; it does not manufacture yaw targets.
    """
    architecture = "dong_probabilistic_bev"
    requires_radar = False
    head_type = "probabilistic_anchor"
    fidelity = "Dong2020 published architecture/loss reconstruction; explicit width,input,assignment adaptation"

    def __init__(self, grid=None, base_channels=64, anchors=None, anchor_angles=None, oriented=False, **kwargs):
        super().__init__()
        self.grid = grid or BEVGrid()
        if base_channels < 2:
            raise ValueError("base_channels must be >=2")
        self.oriented = bool(oriented)
        c = base_channels
        self.stem = conv_bn(3, c, 7, 2)
        self.encoder = nn.ModuleList()
        previous = c
        for stage, out in enumerate((c, 2*c, 4*c, 8*c)):
            self.encoder.append(nn.Sequential(BasicResNetBlock(previous, out, 2), BasicResNetBlock(out, out)))
            previous = out
        lateral_channels = (4*c, 2*c, c, c)
        self.lateral = nn.ModuleList(nn.Conv2d(ch, c, 1) for ch in lateral_channels)
        self.deep = nn.Conv2d(8*c, c, 1)
        self.decoder = nn.ModuleList(nn.Sequential(conv_bn(2*c, c), conv_bn(c, c)) for _ in range(4))
        self.neck = nn.Sequential(conv_bn(c, c), conv_bn(c, c))
        # The paper's 15x42-pixel anchors are tied to its own projection scale.
        # Here defaults express a disclosed 2m x4.5m vehicle; fit training anchors
        # on the target dataset, do not transfer pixels across different grids.
        anchors = [(2/(self.grid.x_max-self.grid.x_min), 4.5/(self.grid.y_max-self.grid.y_min))]*3 if anchors is None else anchors
        angles = [math.radians(a) for a in (30,115,136)] if anchor_angles is None else anchor_angles
        anchors = torch.as_tensor(anchors, dtype=torch.float32)
        angles = torch.as_tensor(angles, dtype=torch.float32)
        if anchors.shape != (3,2) or angles.shape != (3,) or not torch.isfinite(anchors).all() or not torch.isfinite(angles).all() or (anchors <= 0).any():
            raise ValueError("Probabilistic model requires three positive anchor sizes and three finite angles")
        self.register_buffer("anchors", anchors)
        self.register_buffer("anchor_angles", angles)
        self.head = nn.Conv2d(c, 3*13, 1)
        self.export_config = {"name": "probabilistic", "grid": self.grid.to_dict(), "base_channels": base_channels,
                              "anchors": anchors.tolist(), "anchor_angles": angles.tolist(), "oriented": self.oriented}
        self.apply(_xavier)
        with torch.no_grad():
            bias = self.head.bias.reshape(3, 13)
            bias[:,0] = -2.19
            bias[:,5] = 1 # cos(0) initialization

    def forward(self, image, radar=None):
        if image.ndim != 4 or image.shape[1] != 3 or min(image.shape[-2:]) < 16:
            raise ValueError("Probabilistic model requires B,3,H,W Cartesian input with H,W>=16")
        x = self.stem(image)
        skips = [x]
        for stage in self.encoder:
            x = stage(x)
            skips.append(x)
        x = self.deep(x)
        for up, lateral, skip in zip(self.decoder, self.lateral, reversed(skips[:-1])):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = up(torch.cat([x, lateral(skip)], dim=1))
        x = self.neck(x)
        raw = self.head(x).reshape(len(x), 3, 13, *x.shape[-2:]).permute(0,3,4,1,2)
        return {"head_type": self.head_type, "raw": raw, "anchors": self.anchors,
                "anchor_angles": self.anchor_angles, "oriented": self.oriented}


def create_model(name="unet", grid=None, **config):
    allowed = {"base_channels","input_shape","anchors","repeats","upstream_size_clamp","anchor_angles","oriented"}
    unknown = set(config)-allowed
    if unknown:
        raise ValueError(f"Unknown BEV model settings: {sorted(unknown)}")
    grid = BEVGrid.from_dict(grid) if isinstance(grid, dict) else grid or BEVGrid()
    if name in ("unet", "coordinate_unet_bev"):
        if config.get("oriented", False):
            raise ValueError("Coordinate U-Net currently predicts axis-aligned Cartesian boxes")
        return CartesianUNet(grid=grid, **config)
    if name in ("unet_zhang", "zhang_presence_unet_bev"):
        if config.get("oriented", False):
            raise ValueError("Zhang U-Net predicts axis-aligned Cartesian boxes")
        return ZhangUNet(grid=grid, **config)
    if name in ("raddet", "raddet_cartesian_port"):
        if config.get("oriented", False):
            raise ValueError("Released RADDet Cartesian head predicts axis-aligned boxes")
        return RADDetBEV(grid=grid, **config)
    if name in ("probabilistic", "dong_probabilistic_bev"):
        return ProbabilisticBEV(grid=grid, **config)
    raise ValueError(f"Unknown BEV model {name!r}")
