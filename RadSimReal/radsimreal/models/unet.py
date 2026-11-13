"""U-Net-family reconstruction, RadSimReal Sec. 4 and Zhang et al. Sec. 4.3.

Four pooling/transpose-convolution levels and skip connections follow the cited
architecture family. Layer widths, GroupNorm, center heatmap, fractional offsets
and box-size heads are implementation choices because the RadSimReal adaptation
was not released. The input is [fixed-normalized log intensity, Cartesian x, y].
The two Cartesian channels are supplied by the dataset, rather than inferred
from image indices. No proprietary data, torchvision or weights are required.
"""

import torch
from torch import nn
from torch.nn import functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


class Up(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        self.conv = DoubleConv(out_channels * 2, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat((skip, x), dim=1))


class CoordinateUNet(nn.Module):
    """Single-class car detector reconstruction; not an exact RadSimReal model.

    Returns heatmap logits Bx1xHxW, offset Bx2xHxW and size Bx2xHxW.
    Offset order is (x-column, y-row), each in [0,1] within a grid cell;
    size order is (width, height), normalized to the complete image dimensions.
    One center per pixel is supported. Target collisions are counted explicitly.
    """

    fidelity = "reconstructed_unet_family_with_unpublished_heads_approximated"

    def __init__(self, base_channels: int = 16, in_channels: int = 3):
        super().__init__()
        if base_channels < 2 or in_channels < 1:
            raise ValueError("base_channels must be >=2 and in_channels >=1")
        self.base_channels = int(base_channels)
        self.in_channels = int(in_channels)
        channels = [base_channels * (2 ** i) for i in range(5)]
        self.stem = DoubleConv(in_channels, channels[0])
        self.pool = nn.MaxPool2d(2)
        self.down = nn.ModuleList(DoubleConv(channels[i], channels[i + 1]) for i in range(4))
        self.up = nn.ModuleList(Up(channels[i], channels[i - 1]) for i in range(4, 0, -1))
        self.refine = DoubleConv(channels[0], channels[0])
        self.heatmap_head = nn.Conv2d(channels[0], 1, 1)
        self.offset_head = nn.Conv2d(channels[0], 2, 1)
        self.size_head = nn.Conv2d(channels[0], 2, 1)
        nn.init.constant_(self.heatmap_head.bias, -2.19)
        nn.init.constant_(self.size_head.bias, -2.2)

    def forward(self, image):
        if image.ndim != 4 or image.shape[1] != self.in_channels:
            raise ValueError(f"expected Bx{self.in_channels}xHxW input")
        if min(image.shape[-2:]) < 16:
            raise ValueError("four-level U-Net requires image dimensions >=16")
        x = self.stem(image)
        skips = [x]
        for down in self.down:
            x = down(self.pool(x))
            skips.append(x)
        for up, skip in zip(self.up, reversed(skips[:-1])):
            x = up(x, skip)
        x = self.refine(x)
        return {
            "heatmap": self.heatmap_head(x),
            "offset": self.offset_head(x).sigmoid(),
            "size": self.size_head(x).sigmoid(),
        }
