"""
se_resnet3d.py

3D adaptation of the SE-ResNet50 architecture from moskomule/senet.pytorch
(https://github.com/moskomule/senet.pytorch), inflated from 2D convolutions
to 3D convolutions for volumetric CT patch input, with a multi-head sigmoid
output for the 10 CIR radiological characteristic confidence scores.

Reference layer layout (confirmed from moskomule/senet.pytorch/senet/se_resnet.py):
    se_resnet50: ResNet(SEBottleneck, [3, 4, 6, 3])

This is an independent reimplementation in 3D; no pretrained 2D weights are
loaded or inflated, since ImageNet RGB weights have no valid initialization
for single-channel volumetric HU input.
"""
from typing import Dict, List, Optional

import torch
import torch.nn as nn


class SELayer3D(nn.Module):
    """Squeeze-and-Excitation gating module, 3D version.

    Mirrors moskomule/senet.pytorch's SELayer: global average pool -> two
    FC layers with a bottleneck reduction -> sigmoid gate -> channel-wise
    rescale of the input feature map.
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _, _ = x.shape
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)


class SEBottleneck3D(nn.Module):
    """3D bottleneck residual block with an SE gate before the residual add.

    Channel layout (expansion=4) mirrors torchvision/senet.pytorch's
    Bottleneck: 1x1 reduce -> 3x3 spatial -> 1x1 expand, with stride applied
    on the 3x3 convolution (torchvision-style placement, matching
    moskomule's se_resnet50 which subclasses torchvision's ResNet).
    """

    expansion = 4

    def __init__(self, in_channels: int, planes: int, stride: int = 1,
                 downsample: Optional[nn.Module] = None, reduction: int = 16):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride,
                                padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer3D(planes * self.expansion, reduction=reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        return self.relu(out)


class SEResNet3DBackbone(nn.Module):
    """3D SE-ResNet50 trunk: stem + layer1..layer4 + global average pool.

    `layer4` is exposed as a plain attribute (not buried in a Sequential
    wrapper) specifically so Grad-CAM hooks in cir_multihead_pipeline.py and
    inference_cpu.py can attach via getattr(model, 'layer4').
    """

    def __init__(self, in_channels: int = 1, layers: List[int] = (3, 4, 6, 3),
                 reduction: int = 16):
        super().__init__()
        self.inplanes = 64

        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
        )

        self.layer1 = self._make_layer(64, layers[0], stride=1, reduction=reduction)
        self.layer2 = self._make_layer(128, layers[1], stride=2, reduction=reduction)
        self.layer3 = self._make_layer(256, layers[2], stride=2, reduction=reduction)
        self.layer4 = self._make_layer(512, layers[3], stride=2, reduction=reduction)

        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.out_channels = 512 * SEBottleneck3D.expansion

        self._init_weights()

    def _make_layer(self, planes: int, blocks: int, stride: int, reduction: int) -> nn.Sequential:
        downsample = None
        out_channels = planes * SEBottleneck3D.expansion
        if stride != 1 or self.inplanes != out_channels:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels),
            )

        layers = [SEBottleneck3D(self.inplanes, planes, stride=stride,
                                  downsample=downsample, reduction=reduction)]
        self.inplanes = out_channels
        for _ in range(1, blocks):
            layers.append(SEBottleneck3D(self.inplanes, planes, reduction=reduction))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)


class MultiHeadSEResNet3D(nn.Module):
    """Shared SE-ResNet3D trunk with one independent linear+sigmoid head per
    radiological characteristic.

    forward(x) returns a dict[str, Tensor] mapping head name -> confidence
    score tensor of shape (N,) in [0, 1], matching the interface documented
    in cir_multihead_pipeline.create_multihead_model.
    """

    def __init__(self, in_channels: int = 1, head_names: Optional[List[str]] = None,
                 layers: List[int] = (3, 4, 6, 3), reduction: int = 16):
        super().__init__()
        self.head_names = list(head_names) if head_names else []
        if not self.head_names:
            raise ValueError('head_names must be a non-empty list of feature names')

        self.backbone = SEResNet3DBackbone(in_channels=in_channels, layers=layers,
                                            reduction=reduction)
        # Expose layer4 directly on this module too, so Grad-CAM hooks can
        # attach via getattr(model, 'layer4') without reaching into .backbone.
        self.layer4 = self.backbone.layer4

        feat_dim = self.backbone.out_channels
        self.heads = nn.ModuleDict({
            name: nn.Linear(feat_dim, 1) for name in self.head_names
        })

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.backbone(x)
        outputs = {}
        for name, head in self.heads.items():
            logit = head(features).squeeze(-1)
            outputs[name] = torch.sigmoid(logit)
        return outputs


def se_resnet50_3d(in_channels: int = 1, head_names: Optional[List[str]] = None,
                    reduction: int = 16) -> MultiHeadSEResNet3D:
    """Construct a 3D SE-ResNet50 multi-head model: [3, 4, 6, 3] SEBottleneck3D
    blocks per stage, matching moskomule/senet.pytorch's se_resnet50 layout.
    """
    return MultiHeadSEResNet3D(in_channels=in_channels, head_names=head_names,
                                layers=[3, 4, 6, 3], reduction=reduction)


if __name__ == '__main__':
    names = ['spiculation', 'lobulation', 'density', 'calcification', 'margin',
             'texture', 'sphericity', 'subtlety', 'internalStructure', 'malignancy']
    model = se_resnet50_3d(in_channels=1, head_names=names)
    x = torch.randn(2, 1, 64, 64, 64)
    out = model(x)
    for k, v in out.items():
        print(k, v.shape, v.min().item(), v.max().item())