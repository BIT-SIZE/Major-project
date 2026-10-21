import torch.nn as nn
from ml.models.unet import UNet
from ml.models.deeplabv3plus import DeepLabV3Plus
from ml.models.segformer import SegFormerB0

def build_model(architecture: str, num_classes: int = 1, **kwargs) -> nn.Module:
    arch = architecture.lower().replace("-", "").replace("_", "")
    if arch == "unet":
        base_features = kwargs.get("base_features", 32)
        return UNet(n_channels=3, n_classes=num_classes, base_features=base_features)
    elif "deeplab" in arch:
        pretrained = kwargs.get("pretrained_backbone", True)
        freeze = kwargs.get("freeze_backbone", False)
        return DeepLabV3Plus(num_classes=num_classes, pretrained_backbone=pretrained, freeze_backbone=freeze)
    elif "segformer" in arch:
        return SegFormerB0(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
