import os
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from ml.evaluation.metrics import SegmentationMetricsTracker
from ml.training.checkpoint import save_checkpoint

class SegmentationTrainer:
    """
    Standardized trainer for segmentation models across PlantSeg splits.
    Includes training & validation loops, metric logging, checkpointing, and curves generation.
    """
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        device: Optional[torch.device] = None,
        num_epochs: int = 10,
        model_name: str = "unet",
        checkpoint_dir: str = "models/checkpoints",
        reports_dir: str = "reports/training",
        early_stopping_patience: int = 5
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_epochs = num_epochs
        self.model_name = model_name
        self.checkpoint_dir = checkpoint_dir
        self.reports_dir = reports_dir
        self.early_stopping_patience = early_stopping_patience

        self.model.to(self.device)

        self.history = {
            "train_loss": [],
            "val_loss": [],
            "val_iou": [],
            "val_dice": [],
            "val_precision": [],
            "val_recall": [],
            "learning_rate": []
        }
        self.best_dice = -1.0
        self.patience_counter = 0

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        running_loss = 0.0
        total_batches = len(self.train_loader)

        for batch_idx, (images, masks, _) in enumerate(self.train_loader):
            images = images.to(self.device)
            masks = masks.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(images)
            loss = self.criterion(logits, masks)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

            if (batch_idx + 1) % max(1, total_batches // 4) == 0 or (batch_idx + 1) == total_batches:
                print(f"Epoch [{epoch+1}/{self.num_epochs}] Step [{batch_idx+1}/{total_batches}] Batch Loss: {loss.item():.4f}")

        return running_loss / max(1, total_batches)

    @torch.no_grad()
    def validate_epoch(self) -> Dict[str, float]:
        self.model.eval()
        running_loss = 0.0
        metrics_tracker = SegmentationMetricsTracker(threshold=0.5)

        for images, masks, _ in self.val_loader:
            images = images.to(self.device)
            masks = masks.to(self.device)

            logits = self.model(images)
            loss = self.criterion(logits, masks)
            running_loss += loss.item()

            metrics_tracker.update(logits, masks)

        val_metrics = metrics_tracker.compute()
        val_metrics["val_loss"] = round(running_loss / max(1, len(self.val_loader)), 4)
        return val_metrics

    def fit(self) -> Dict[str, Any]:
        print(f"\n================ STARTING TRAINING: {self.model_name.upper()} ================")
        print(f"Device: {self.device} | Epochs: {self.num_epochs} | Train Batches: {len(self.train_loader)} | Val Batches: {len(self.val_loader)}")
        start_time = time.time()

        for epoch in range(self.num_epochs):
            t0 = time.time()
            train_loss = self.train_epoch(epoch)
            val_metrics = self.validate_epoch()

            current_lr = self.optimizer.param_groups[0]["lr"]
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics["val_loss"])
                else:
                    self.scheduler.step()

            # Record metrics
            self.history["train_loss"].append(round(train_loss, 4))
            self.history["val_loss"].append(val_metrics["val_loss"])
            self.history["val_iou"].append(val_metrics["mean_iou"])
            self.history["val_dice"].append(val_metrics["mean_dice"])
            self.history["val_precision"].append(val_metrics["mean_precision"])
            self.history["val_recall"].append(val_metrics["mean_recall"])
            self.history["learning_rate"].append(current_lr)

            epoch_time = time.time() - t0
            print(f"[Epoch {epoch+1}/{self.num_epochs} - {epoch_time:.1f}s] Train Loss: {train_loss:.4f} | Val Loss: {val_metrics['val_loss']} | Val Dice: {val_metrics['mean_dice']} | Val IoU: {val_metrics['mean_iou']}")

            # Checkpoint
            is_best = val_metrics["mean_dice"] > self.best_dice
            if is_best:
                self.best_dice = val_metrics["mean_dice"]
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            state = {
                "epoch": epoch + 1,
                "model_name": self.model_name,
                "state_dict": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "val_dice": val_metrics["mean_dice"],
                "val_iou": val_metrics["mean_iou"],
                "val_loss": val_metrics["val_loss"]
            }
            save_checkpoint(state, is_best, checkpoint_dir=self.checkpoint_dir, model_name=self.model_name)

            if self.patience_counter >= self.early_stopping_patience:
                print(f"[EARLY STOPPING] Patience of {self.early_stopping_patience} epochs reached.")
                break

        total_duration = time.time() - start_time
        print(f"Training completed in {total_duration:.1f}s. Best Val Dice: {self.best_dice:.4f}")

        # Save logs and curves
        self._save_training_artifacts(total_duration)
        return self.history

    def _save_training_artifacts(self, duration: float):
        os.makedirs(self.reports_dir, exist_ok=True)
        # Save JSON history
        history_path = Path(self.reports_dir) / f"{self.model_name}_training_history.json"
        summary = {
            "model_name": self.model_name,
            "training_duration_seconds": round(duration, 2),
            "best_val_dice": self.best_dice,
            "final_metrics": {k: v[-1] for k, v in self.history.items() if v},
            "history": self.history
        }
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        # Plot curves
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        epochs = range(1, len(self.history["train_loss"]) + 1)
        axes[0].plot(epochs, self.history["train_loss"], label="Train Loss", color="#d9534f")
        axes[0].plot(epochs, self.history["val_loss"], label="Val Loss", color="#0275d8")
        axes[0].set_title(f"{self.model_name.upper()} - Loss Curves")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].legend()
        axes[0].grid(True, linestyle="--", alpha=0.6)

        axes[1].plot(epochs, self.history["val_dice"], label="Val Dice", color="#5cb85c")
        axes[1].plot(epochs, self.history["val_iou"], label="Val IoU", color="#f0ad4e")
        axes[1].set_title(f"{self.model_name.upper()} - Segmentation Performance")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Metric Score")
        axes[1].legend()
        axes[1].grid(True, linestyle="--", alpha=0.6)

        plt.tight_layout()
        curve_path = Path(self.reports_dir) / f"{self.model_name}_training_curves.png"
        plt.savefig(curve_path, dpi=120)
        plt.close()
        print(f"[REPORTS] Saved training curve figure to {curve_path}")
