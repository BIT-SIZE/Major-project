import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class ASPPConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, dilation: int):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        super().__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels: int, atrous_rates: list = [6, 12, 18], out_channels: int = 256):
        super().__init__()
        modules = [
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        ]
        for rate in atrous_rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)
        self.project = nn.Sequential(
            nn.Conv2d(len(modules) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


class DeepLabV3Plus(nn.Module):
    """
    DeepLabV3+ with ResNet-50 backbone.
    Combines Atrous Spatial Pyramid Pooling (ASPP) with low-level skip features
    for precise boundary delineation of disease lesions.
    """
    def __init__(
        self,
        num_classes: int = 1,
        pretrained_backbone: bool = True,
        freeze_backbone: bool = False
    ):
        super().__init__()
        # Load ResNet-50
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained_backbone else None)

        self.initial = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )
        self.layer1 = resnet.layer1  # Low-level features: [B, 256, H/4, W/4]
        self.layer2 = resnet.layer2  # [B, 512, H/8, W/8]
        self.layer3 = resnet.layer3  # [B, 1024, H/16, W/16]
        self.layer4 = resnet.layer4  # High-level features: [B, 2048, H/32, W/32]

        if freeze_backbone:
            for p in self.initial.parameters():
                p.requires_grad = False
            for p in self.layer1.parameters():
                p.requires_grad = False
            for p in self.layer2.parameters():
                p.requires_grad = False
            for p in self.layer3.parameters():
                p.requires_grad = False
            for p in self.layer4.parameters():
                p.requires_grad = False

        self.aspp = ASPP(in_channels=2048, atrous_rates=[6, 12, 18], out_channels=256)

        # Low-level projection
        self.low_level_proj = nn.Sequential(
            nn.Conv2d(256, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )

        # Decoder head
        self.decoder = nn.Sequential(
            nn.Conv2d(256 + 48, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(256, num_classes, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]

        # Backbone pass
        x0 = self.initial(x)
        low_level = self.layer1(x0)
        x2 = self.layer2(low_level)
        x3 = self.layer3(x2)
        high_level = self.layer4(x3)

        # ASPP high-level pass
        x_aspp = self.aspp(high_level)
        x_aspp_up = F.interpolate(x_aspp, size=low_level.shape[-2:], mode="bilinear", align_corners=False)

        # Low level projection
        low_proj = self.low_level_proj(low_level)

        # Concat & decode
        combined = torch.cat([x_aspp_up, low_proj], dim=1)
        logits_low = self.decoder(combined)

        # Final upsample to original resolution
        logits = F.interpolate(logits_low, size=input_size, mode="bilinear", align_corners=False)
        return logits
