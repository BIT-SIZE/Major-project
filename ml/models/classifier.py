from typing import Tuple, Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class PlantDiseaseClassifier(nn.Module):
    """
    EfficientNet-B0 plant disease classifier.
    Fine-tunes ImageNet pretrained weights to predict 114 fine-grained foliar disease classes.
    """
    def __init__(
        self,
        num_classes: int = 114,
        pretrained: bool = True,
        dropout: float = 0.3
    ):
        super().__init__()
        self.num_classes = num_classes

        # Load EfficientNet-B0 with ImageNet pretrained weights
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.backbone = models.efficientnet_b0(weights=weights)

        # Replace classification head
        in_features = self.backbone.classifier[1].in_features  # 1280
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Returns raw unnormalized logits: [B, num_classes]
        return self.backbone(x)

    @torch.no_grad()
    def predict_top_k(self, x: torch.Tensor, k: int = 3) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns top-k probability scores and class indices.
        x: [B, 3, 224, 224]
        returns: (top_scores, top_indices)
        """
        self.eval()
        logits = self.forward(x)
        probs = F.softmax(logits, dim=1)
        top_scores, top_indices = torch.topk(probs, k=min(k, self.num_classes), dim=1)
        return top_scores, top_indices
