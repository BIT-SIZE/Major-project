from typing import Dict, Any, Union
import numpy as np
import torch

class SegmentationMetricsTracker:
    """
    Computes and aggregates IoU, Dice/F1, Precision, and Recall for binary masks.
    Safely handles edge cases like empty ground-truth masks.
    """
    def __init__(self, threshold: float = 0.5, eps: float = 1e-7):
        self.threshold = threshold
        self.eps = eps
        self.reset()

    def reset(self):
        self.total_tp = 0.0
        self.total_fp = 0.0
        self.total_fn = 0.0
        self.total_tn = 0.0
        self.sample_ious = []
        self.sample_dices = []
        self.sample_precisions = []
        self.sample_recalls = []

    def update(self, preds: Union[torch.Tensor, np.ndarray], targets: Union[torch.Tensor, np.ndarray]):
        if isinstance(preds, torch.Tensor):
            preds = preds.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()

        # If logits, apply sigmoid
        if preds.min() < 0.0 or preds.max() > 1.0:
            preds = 1.0 / (1.0 + np.exp(-preds))

        binary_preds = (preds >= self.threshold).astype(np.float32)
        binary_targets = (targets >= 0.5).astype(np.float32)

        # Batch iteration
        for p, t in zip(binary_preds, binary_targets):
            tp = float(np.sum((p == 1.0) & (t == 1.0)))
            fp = float(np.sum((p == 1.0) & (t == 0.0)))
            fn = float(np.sum((p == 0.0) & (t == 1.0)))
            tn = float(np.sum((p == 0.0) & (t == 0.0)))

            self.total_tp += tp
            self.total_fp += fp
            self.total_fn += fn
            self.total_tn += tn

            denom_iou = tp + fp + fn
            iou = (tp + self.eps) / (denom_iou + self.eps) if denom_iou > 0 else (1.0 if (tp == 0 and fp == 0 and fn == 0) else 0.0)

            denom_dice = (2.0 * tp) + fp + fn
            dice = ((2.0 * tp) + self.eps) / (denom_dice + self.eps) if denom_dice > 0 else (1.0 if (tp == 0 and fp == 0 and fn == 0) else 0.0)

            prec = (tp + self.eps) / (tp + fp + self.eps) if (tp + fp) > 0 else 0.0
            rec = (tp + self.eps) / (tp + fn + self.eps) if (tp + fn) > 0 else 0.0

            self.sample_ious.append(iou)
            self.sample_dices.append(dice)
            self.sample_precisions.append(prec)
            self.sample_recalls.append(rec)

    def compute(self) -> Dict[str, float]:
        mean_iou = float(np.mean(self.sample_ious)) if self.sample_ious else 0.0
        mean_dice = float(np.mean(self.sample_dices)) if self.sample_dices else 0.0
        mean_prec = float(np.mean(self.sample_precisions)) if self.sample_precisions else 0.0
        mean_rec = float(np.mean(self.sample_recalls)) if self.sample_recalls else 0.0

        # Global dataset-level metrics
        global_iou = (self.total_tp + self.eps) / (self.total_tp + self.total_fp + self.total_fn + self.eps)
        global_dice = (2.0 * self.total_tp + self.eps) / (2.0 * self.total_tp + self.total_fp + self.total_fn + self.eps)
        global_prec = (self.total_tp + self.eps) / (self.total_tp + self.total_fp + self.eps)
        global_rec = (self.total_tp + self.eps) / (self.total_tp + self.total_fn + self.eps)

        return {
            "mean_iou": round(mean_iou, 4),
            "mean_dice": round(mean_dice, 4),
            "mean_precision": round(mean_prec, 4),
            "mean_recall": round(mean_rec, 4),
            "global_iou": round(float(global_iou), 4),
            "global_dice": round(float(global_dice), 4),
            "global_precision": round(float(global_prec), 4),
            "global_recall": round(float(global_rec), 4)
        }
