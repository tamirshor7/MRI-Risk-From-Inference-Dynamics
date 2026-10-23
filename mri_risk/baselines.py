import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True), nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.net(x)


class ErrorUNet(nn.Module):

    def __init__(self, base=24):
        super().__init__()
        self.e1 = ConvBlock(1, base)
        self.e2 = ConvBlock(base, base * 2)
        self.e3 = ConvBlock(base * 2, base * 4)
        self.e4 = ConvBlock(base * 4, base * 8)
        self.mid = ConvBlock(base * 8, base * 16)
        self.d4 = ConvBlock(base * 16 + base * 8, base * 8)
        self.d3 = ConvBlock(base * 8 + base * 4, base * 4)
        self.d2 = ConvBlock(base * 4 + base * 2, base * 2)
        self.d1 = ConvBlock(base * 2 + base, base)
        self.out = nn.Conv2d(base, 1, kernel_size=1)

    def down(self, x):
        return F.max_pool2d(x, 2)

    def upcat(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        return torch.cat([x, skip], dim=1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.down(e1))
        e3 = self.e3(self.down(e2))
        e4 = self.e4(self.down(e3))
        m = self.mid(self.down(e4))
        d4 = self.d4(self.upcat(m, e4))
        d3 = self.d3(self.upcat(d4, e3))
        d2 = self.d2(self.upcat(d3, e2))
        d1 = self.d1(self.upcat(d2, e1))
        return F.softplus(self.out(d1))


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(out_ch))
        else:
            self.downsample = None

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(identity)
        out = self.relu(out + identity)
        return out


class ResNet18Risk(nn.Module):

    def __init__(self):
        super().__init__()
        self.in_ch = 64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._layer(64, 2, 1)
        self.layer2 = self._layer(128, 2, 2)
        self.layer3 = self._layer(256, 2, 2)
        self.layer4 = self._layer(512, 2, 2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 1)

    def _layer(self, out_ch, blocks, stride):
        layers = [BasicBlock(self.in_ch, out_ch, stride)]
        self.in_ch = out_ch
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.in_ch, out_ch, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x).squeeze(1)


class RMModel(nn.Module):

    def __init__(self, im2im_root):
        super().__init__()
        sys.path.insert(0, str(Path(im2im_root).resolve(strict=True)))
        from core.models.trunks.unet import UNet
        from core.models.finallayers.residual_magnitude_layer import ResidualMagnitudeLayer
        self.trunk = UNet(n_channels_in=1, n_channels_out=1)
        self.head = ResidualMagnitudeLayer(n_channels_middle=32, n_channels_out=1, params={})

    def forward(self, x):
        return self.head(self.trunk(x))

def make_cqr(giannakopoulos_root):
    root = Path(giannakopoulos_root).resolve(strict=True)
    sys.path.insert(0, str(root))
    from varnets.VarNet import UncertaintyNetwork
    return UncertaintyNetwork(
        head_chans=32,
        head_pools=4,
        drop_prob=0.0,
    )
