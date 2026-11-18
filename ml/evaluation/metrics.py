from typing import Dict, Any, Union, Optional
import numpy as np
import torch

class SegmentationMetricsTracker:
    """
    Computes and aggregates IoU, Dice/F1, Precision, and Recall for binary lesion segmentation.
    Explicitly handles raw logits by applying sigmoid without relying on heuristic value checks.
    Safely handles edge cases like empty ground-truth masks without numerical instability.
    Excludes true negatives (background pixels) from IoU and Dice calculations.
    """
    def __init__(self, threshold: float = 0.5, from_logits: bool = True, eps: float = 1e-7):
        self.threshold = threshold
        self.from_logits = from_logits
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

    def update(
        self,
        preds: Union[torch.Tensor, np.ndarray],
        targets: Union[torch.Tensor, np.ndarray],
        from_logits: Optional[bool] = None
    ):
        """
        Updates running confusion statistics with a batch of predictions and ground-truth masks.
        """
        # Determine whether to apply sigmoid based on explicit argument or instance default
        apply_sigmoid = self.from_logits if from_logits is None else from_logits

        if apply_sigmoid:
            if isinstance(preds, torch.Tensor):
                probs = torch.sigmoid(preds)
            else:
                probs = 1.0 / (1.0 + np.exp(-np.asarray(preds, dtype=np.float64)))
        else:
            probs = preds

        # Convert to numpy arrays for calculation
        if isinstance(probs, torch.Tensor):
            probs = probs.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()

        binary_preds = (probs >= self.threshold).astype(np.float32)
        binary_targets = (targets >= 0.5).astype(np.float32)

        # Iterate through batch samples
        for p, t in zip(binary_preds, binary_targets):
            p_flat = p.reshape(-1)
            t_flat = t.reshape(-1)

            tp = float(np.sum((p_flat == 1.0) & (t_flat == 1.0)))
            fp = float(np.sum((p_flat == 1.0) & (t_flat == 0.0)))
            fn = float(np.sum((p_flat == 0.0) & (t_flat == 1.0)))
            tn = float(np.sum((p_flat == 0.0) & (t_flat == 0.0)))

            self.total_tp += tp
            self.total_fp += fp
            self.total_fn += fn
            self.total_tn += tn

            target_has_lesion = (tp + fn > 0)
            pred_has_lesion = (tp + fp > 0)

            if not target_has_lesion:
                # Ground truth has no lesion (healthy leaf)
                if not pred_has_lesion:
                    # Model correctly predicted zero lesion pixels
                    iou = 1.0
                    dice = 1.0
                    prec = 1.0
                    rec = 1.0
                else:
                    # Model falsely predicted lesions on healthy leaf
                    iou = 0.0
                    dice = 0.0
                    prec = 0.0
                    rec = 0.0
            else:
                # Ground truth has lesion pixels
                if tp == 0:
                    iou = 0.0
                    dice = 0.0
                    prec = 0.0
                    rec = 0.0
                else:
                    denom_iou = tp + fp + fn
                    iou = (tp / denom_iou) if denom_iou > 0 else 0.0

                    denom_dice = (2.0 * tp) + fp + fn
                    dice = ((2.0 * tp) / denom_dice) if denom_dice > 0 else 0.0

                    denom_prec = tp + fp
                    prec = (tp / denom_prec) if denom_prec > 0 else 0.0

                    denom_rec = tp + fn
                    rec = (tp / denom_rec) if denom_rec > 0 else 0.0

            self.sample_ious.append(iou)
            self.sample_dices.append(dice)
            self.sample_precisions.append(prec)
            self.sample_recalls.append(rec)

    def compute(self) -> Dict[str, float]:
        """
        Computes both macro-averaged (per-image) and dataset-level pooled (global) metrics.
        """
        mean_iou = float(np.mean(self.sample_ious)) if self.sample_ious else 0.0
        mean_dice = float(np.mean(self.sample_dices)) if self.sample_dices else 0.0
        mean_prec = float(np.mean(self.sample_precisions)) if self.sample_precisions else 0.0
        mean_rec = float(np.mean(self.sample_recalls)) if self.sample_recalls else 0.0

        # Dataset-level pooled global metrics
        total_positives = self.total_tp + self.total_fn
        total_predicted = self.total_tp + self.total_fp

        if total_positives == 0 and total_predicted == 0:
            # Entire evaluated dataset has no positive lesion pixels and no false-positive predictions
            global_iou = 1.0
            global_dice = 1.0
            global_prec = 1.0
            global_rec = 1.0
        elif total_positives == 0 and total_predicted > 0:
            # Entire dataset has no lesions, but model predicted false positives
            global_iou = 0.0
            global_dice = 0.0
            global_prec = 0.0
            global_rec = 0.0
        elif self.total_tp == 0:
            # Dataset has ground truth lesions, but model predicted zero true positives
            global_iou = 0.0
            global_dice = 0.0
            global_prec = 0.0
            global_rec = 0.0
        else:
            denom_global_iou = self.total_tp + self.total_fp + self.total_fn
            global_iou = float(self.total_tp / denom_global_iou) if denom_global_iou > 0 else 0.0

            denom_global_dice = (2.0 * self.total_tp) + self.total_fp + self.total_fn
            global_dice = float((2.0 * self.total_tp) / denom_global_dice) if denom_global_dice > 0 else 0.0

            denom_global_prec = self.total_tp + self.total_fp
            global_prec = float(self.total_tp / denom_global_prec) if denom_global_prec > 0 else 0.0

            denom_global_rec = self.total_tp + self.total_fn
            global_rec = float(self.total_tp / denom_global_rec) if denom_global_rec > 0 else 0.0

        return {
            "mean_iou": float(mean_iou),
            "mean_dice": float(mean_dice),
            "mean_precision": float(mean_prec),
            "mean_recall": float(mean_rec),
            "global_iou": float(global_iou),
            "global_dice": float(global_dice),
            "global_precision": float(global_prec),
            "global_recall": float(global_rec)
        }
