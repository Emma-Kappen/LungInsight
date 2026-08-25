"""
se_resnet3d.py

3D SE-ResNet50 for volumetric CT patches.

Architecture
------------
Input:
    (N, C, Z, Y, X)

For LungInsight:
    (N, 1, 64, 64, 64)

Backbone:
    Stem
    -> layer1
    -> layer2
    -> layer3
    -> layer4
    -> global average pooling

Output:
    Dictionary of independent scalar regression predictions:

        {
            "calcification": Tensor(N,),
            "lobulation": Tensor(N,),
            ...
        }

Training contract
-----------------
The current training pipeline uses masked MSE directly on the raw outputs:

    MSE(raw_prediction, target)

Therefore the model outputs are regression scores, not probabilities and not
BCE logits. Do NOT apply sigmoid inside this model.

Reference architecture:
    SE-ResNet50 bottleneck layout: [3, 4, 6, 3]

This is an independent 3D implementation adapted for single-channel
volumetric CT input.
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn


class SELayer3D(nn.Module):
    """
    3D Squeeze-and-Excitation channel attention.

    Flow:
        (N, C, Z, Y, X)
            ->
        AdaptiveAvgPool3d(1)
            ->
        FC(C -> C/reduction -> C)
            ->
        sigmoid channel gates
            ->
        channel-wise feature rescaling
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()

        if channels <= 0:
            raise ValueError(
                f"channels must be positive, got {channels}"
            )

        if reduction <= 0:
            raise ValueError(
                f"reduction must be positive, got {reduction}"
            )

        reduced_channels = max(1, channels // reduction)

        self.avg_pool = nn.AdaptiveAvgPool3d(1)

        self.fc = nn.Sequential(
            nn.Linear(
                channels,
                reduced_channels,
                bias=False,
            ),
            nn.ReLU(inplace=True),
            nn.Linear(
                reduced_channels,
                channels,
                bias=False,
            ),
            nn.Sigmoid(),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        if x.ndim != 5:
            raise ValueError(
                "SELayer3D expects input with shape "
                "(N,C,Z,Y,X), "
                f"got {tuple(x.shape)}"
            )

        batch_size, channels, _, _, _ = x.shape

        y = self.avg_pool(x)
        y = y.reshape(batch_size, channels)

        y = self.fc(y)

        y = y.reshape(
            batch_size,
            channels,
            1,
            1,
            1,
        )

        return x * y


class SEBottleneck3D(nn.Module):
    """
    3D SE-ResNet bottleneck block.

    Layout:

        1x1x1 Conv
            ->
        BatchNorm
            ->
        ReLU
            ->
        3x3x3 Conv
            ->
        BatchNorm
            ->
        ReLU
            ->
        1x1x1 Conv
            ->
        BatchNorm
            ->
        SE attention
            ->
        Residual addition
            ->
        ReLU

    The bottleneck expansion is 4, matching ResNet50.
    """

    expansion = 4

    def __init__(
        self,
        in_channels: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        reduction: int = 16,
    ):
        super().__init__()

        if in_channels <= 0:
            raise ValueError(
                f"in_channels must be positive, got {in_channels}"
            )

        if planes <= 0:
            raise ValueError(
                f"planes must be positive, got {planes}"
            )

        self.conv1 = nn.Conv3d(
            in_channels,
            planes,
            kernel_size=1,
            bias=False,
        )

        self.bn1 = nn.BatchNorm3d(planes)

        self.conv2 = nn.Conv3d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )

        self.bn2 = nn.BatchNorm3d(planes)

        self.conv3 = nn.Conv3d(
            planes,
            planes * self.expansion,
            kernel_size=1,
            bias=False,
        )

        self.bn3 = nn.BatchNorm3d(
            planes * self.expansion
        )

        self.relu = nn.ReLU(inplace=True)

        self.se = SELayer3D(
            planes * self.expansion,
            reduction=reduction,
        )

        self.downsample = downsample
        self.stride = stride

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)

        return out


class SEResNet3DBackbone(nn.Module):
    """
    3D SE-ResNet50 feature extractor.

    Canonical Grad-CAM target paths:

        backbone.layer1
        backbone.layer2
        backbone.layer3
        backbone.layer4

    For a 64^3 input, approximate feature map resolutions are:

        Input:     64^3
        Stem:      16^3
        layer1:    16^3
        layer2:     8^3
        layer3:     4^3
        layer4:     2^3

    `layer3` is the default Grad-CAM target because it retains more spatial
    information than layer4 while remaining semantically deep.
    """

    def __init__(
        self,
        in_channels: int = 1,
        layers: List[int] = (3, 4, 6, 3),
        reduction: int = 16,
    ):
        super().__init__()

        if len(layers) != 4:
            raise ValueError(
                "layers must contain exactly four stage sizes, "
                f"got {layers}"
            )

        if any(blocks <= 0 for blocks in layers):
            raise ValueError(
                f"all layer block counts must be positive, got {layers}"
            )

        self.inplanes = 64

        self.stem = nn.Sequential(
            nn.Conv3d(
                in_channels,
                64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            ),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(
                kernel_size=3,
                stride=2,
                padding=1,
            ),
        )

        self.layer1 = self._make_layer(
            planes=64,
            blocks=layers[0],
            stride=1,
            reduction=reduction,
        )

        self.layer2 = self._make_layer(
            planes=128,
            blocks=layers[1],
            stride=2,
            reduction=reduction,
        )

        self.layer3 = self._make_layer(
            planes=256,
            blocks=layers[2],
            stride=2,
            reduction=reduction,
        )

        self.layer4 = self._make_layer(
            planes=512,
            blocks=layers[3],
            stride=2,
            reduction=reduction,
        )

        self.avgpool = nn.AdaptiveAvgPool3d(1)

        self.out_channels = (
            512 * SEBottleneck3D.expansion
        )

        self._init_weights()

    def _make_layer(
        self,
        planes: int,
        blocks: int,
        stride: int,
        reduction: int,
    ) -> nn.Sequential:

        out_channels = (
            planes * SEBottleneck3D.expansion
        )

        downsample = None

        if (
            stride != 1
            or self.inplanes != out_channels
        ):
            downsample = nn.Sequential(
                nn.Conv3d(
                    self.inplanes,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm3d(out_channels),
            )

        layers = [
            SEBottleneck3D(
                in_channels=self.inplanes,
                planes=planes,
                stride=stride,
                downsample=downsample,
                reduction=reduction,
            )
        ]

        self.inplanes = out_channels

        for _ in range(1, blocks):
            layers.append(
                SEBottleneck3D(
                    in_channels=self.inplanes,
                    planes=planes,
                    stride=1,
                    downsample=None,
                    reduction=reduction,
                )
            )

        return nn.Sequential(*layers)

    def _init_weights(self) -> None:

        for module in self.modules():

            if isinstance(module, nn.Conv3d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )

            elif isinstance(module, nn.BatchNorm3d):
                nn.init.constant_(
                    module.weight,
                    1,
                )

                nn.init.constant_(
                    module.bias,
                    0,
                )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        x = self.stem(x)

        x = self.layer1(x)

        x = self.layer2(x)

        x = self.layer3(x)

        x = self.layer4(x)

        x = self.avgpool(x)

        return torch.flatten(
            x,
            start_dim=1,
        )


class MultiHeadSEResNet3D(nn.Module):
    """
    Shared 3D SE-ResNet50 backbone with independent scalar regression heads.

    Input:
        (N, 1, Z, Y, X)

    Output:
        {
            feature_name: Tensor(N,)
        }

    The outputs are raw regression predictions.

    Training uses:

        masked MSE(prediction, target)

    Therefore no sigmoid is applied here.
    """

    def __init__(
        self,
        in_channels: int = 1,
        head_names: Optional[List[str]] = None,
        layers: List[int] = (3, 4, 6, 3),
        reduction: int = 16,
    ):
        super().__init__()

        self.head_names = (
            list(head_names)
            if head_names is not None
            else []
        )

        if not self.head_names:
            raise ValueError(
                "head_names must be a non-empty list"
            )

        if len(set(self.head_names)) != len(self.head_names):
            raise ValueError(
                "head_names must not contain duplicates"
            )

        self.backbone = SEResNet3DBackbone(
            in_channels=in_channels,
            layers=layers,
            reduction=reduction,
        )

        feature_dim = self.backbone.out_channels

        self.heads = nn.ModuleDict(
            {
                name: nn.Linear(
                    feature_dim,
                    1,
                )
                for name in self.head_names
            }
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:

        features = self.backbone(x)

        outputs: Dict[str, torch.Tensor] = {}

        for name, head in self.heads.items():

            prediction = head(features).squeeze(-1)

            outputs[name] = prediction

        return outputs


def se_resnet50_3d(
    in_channels: int = 1,
    head_names: Optional[List[str]] = None,
    reduction: int = 16,
) -> MultiHeadSEResNet3D:
    """
    Construct the canonical 3D SE-ResNet50.

    Stage layout:

        layer1: 3 bottleneck blocks
        layer2: 4 bottleneck blocks
        layer3: 6 bottleneck blocks
        layer4: 3 bottleneck blocks
    """

    return MultiHeadSEResNet3D(
        in_channels=in_channels,
        head_names=head_names,
        layers=[3, 4, 6, 3],
        reduction=reduction,
    )


if __name__ == "__main__":

    HEAD_NAMES = [
        "calcification",
        "lobulation",
        "malignancy",
        "margin",
        "sphericity",
        "spiculation",
        "subtlety",
        "texture",
    ]

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = se_resnet50_3d(
        in_channels=1,
        head_names=HEAD_NAMES,
    ).to(device)

    model.eval()

    x = torch.randn(
        2,
        1,
        64,
        64,
        64,
        device=device,
    )

    with torch.no_grad():
        outputs = model(x)

    print("Model output shapes:")

    for name, value in outputs.items():
        print(
            f"{name:16s} "
            f"shape={tuple(value.shape)} "
            f"min={value.min().item():.4f} "
            f"max={value.max().item():.4f}"
        )