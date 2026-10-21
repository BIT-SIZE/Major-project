import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    """
    Computes soft Dice loss for binary segmentation.
    Smooth term prevents division by zero.
    """
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: [B, 1, H, W]
        # targets: [B, 1, H, W]
        probs = torch.sigmoid(logits)
        
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        
        intersection = (probs_flat * targets_flat).sum()
        cardinality = probs_flat.sum() + targets_flat.sum()
        
        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice


class FocalLoss(nn.Module):
    """
    Binary Focal Loss for severe class imbalance.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_t * ((1 - p_t) ** self.gamma) * bce_loss
        return loss.mean()


class CombinedBCEDiceLoss(nn.Module):
    """
    Weighted combination of Binary Cross-Entropy and Dice Loss.
    Standard high-performing objective for lesion segmentation tasks.
    """
    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5, smooth: float = 1.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(smooth=smooth)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss_bce = self.bce(logits, targets)
        loss_dice = self.dice(logits, targets)
        return (self.bce_weight * loss_bce) + (self.dice_weight * loss_dice)


def get_loss_function(name: str = "bce_dice", **kwargs) -> nn.Module:
    name = name.lower()
    if name in ["bce_dice", "combined"]:
        return CombinedBCEDiceLoss(**kwargs)
    elif name == "dice":
        return DiceLoss(**kwargs)
    elif name == "bce":
        return nn.BCEWithLogitsLoss()
    elif name == "focal":
        return FocalLoss(**kwargs)
    else:
        raise ValueError(f"Unsupported loss function: {name}")
