import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    """
    Computes soft Dice loss for binary segmentation using per-image aggregation.
    Per-image aggregation prevents large lesions from dominating small lesions across batches.
    Smooth term ensures numerical stability and handles empty ground-truth masks gracefully.
    """
    def __init__(self, smooth: float = 1.0, eps: float = 1e-7):
        super().__init__()
        self.smooth = smooth
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Align target shape to logits: [B, 1, H, W]
        if targets.ndim == 3:
            targets = targets.unsqueeze(1)
        if targets.shape != logits.shape:
            targets = targets.reshape(logits.shape)

        # Apply sigmoid to convert raw logits to probabilities
        probs = torch.sigmoid(logits)

        # Flatten spatial dimensions per sample: [B, H * W]
        batch_size = probs.size(0)
        probs_flat = probs.reshape(batch_size, -1)
        targets_flat = targets.reshape(batch_size, -1).float()

        intersection = (probs_flat * targets_flat).sum(dim=1)
        cardinality = probs_flat.sum(dim=1) + targets_flat.sum(dim=1)

        # Dice score per image in the batch
        dice_per_image = (2.0 * intersection + self.smooth) / (cardinality + self.smooth + self.eps)
        
        # Loss is 1 - Dice, averaged over batch
        loss = 1.0 - dice_per_image
        return loss.mean()


class FocalLoss(nn.Module):
    """
    Binary Focal Loss for severe class imbalance.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0, eps: float = 1e-7):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if targets.ndim == 3:
            targets = targets.unsqueeze(1)
        if targets.shape != logits.shape:
            targets = targets.reshape(logits.shape)

        targets = targets.float()
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        p_t = torch.clamp(p_t, min=self.eps, max=1.0 - self.eps)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        loss = alpha_t * ((1.0 - p_t) ** self.gamma) * bce_loss
        return loss.mean()


class CombinedBCEDiceLoss(nn.Module):
    """
    Weighted combination of Binary Cross-Entropy with Logits and Per-Image Soft Dice Loss.
    Standard high-performing objective for foliar lesion segmentation tasks.
    """
    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5, smooth: float = 1.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(smooth=smooth)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if targets.ndim == 3:
            targets = targets.unsqueeze(1)
        if targets.shape != logits.shape:
            targets = targets.reshape(logits.shape)

        targets = targets.float()
        loss_bce = self.bce(logits, targets)
        loss_dice = self.dice(logits, targets)
        return (self.bce_weight * loss_bce) + (self.dice_weight * loss_dice)


def get_loss_function(name: str = "bce_dice", **kwargs) -> nn.Module:
    """
    Factory function for instantiation of segmentation loss criteria.
    """
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
