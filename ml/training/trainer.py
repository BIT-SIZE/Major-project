import os
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, Union
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from ml.evaluation.metrics import SegmentationMetricsTracker
from ml.training.checkpoint import save_checkpoint, load_checkpoint

class SegmentationTrainer:
    """
    Standardized trainer for segmentation models across PlantSeg splits.
    Features:
      - Automatic Mixed Precision (AMP) via torch.amp / torch.cuda.amp.
      - Configurable gradient clipping with unscaling.
      - Sample-weighted loss calculations for train and validation.
      - Early stopping with best checkpoint tracking.
      - In-memory model restoration to best checkpoint upon completion.
      - Full state saving and checkpoint resumption (model, optimizer, scheduler, scaler).
      - Training history logging and visual curve plotting.
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
        early_stopping_patience: int = 5,
        use_amp: bool = True,
        grad_clip: Optional[float] = 1.0,
        resume_path: Optional[str] = None,
        min_delta: float = 1e-4
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
        self.grad_clip = grad_clip
        self.min_delta = min_delta

        self.model.to(self.device)

        # Automatic Mixed Precision (AMP) configuration
        self.is_cuda = (self.device.type == "cuda")
        self.use_amp = bool(use_amp and self.is_cuda)
        
        if hasattr(torch.amp, "GradScaler"):
            self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        self.start_epoch = 0
        self.best_dice = -1.0
        self.patience_counter = 0

        self.history = {
            "train_loss": [],
            "val_loss": [],
            "val_iou": [],
            "val_dice": [],
            "val_precision": [],
            "val_recall": [],
            "learning_rate": []
        }

        # Checkpoint resumption if specified
        if resume_path:
            self.resume_from_checkpoint(resume_path)

    def resume_from_checkpoint(self, checkpoint_path: str):
        """
        Resumes training state from a saved checkpoint file.
        Restores model weights, optimizer, scheduler, scaler, epoch, and training history.
        """
        print(f"[RESUME] Loading checkpoint from {checkpoint_path}...")
        checkpoint = load_checkpoint(
            checkpoint_path=checkpoint_path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler if self.use_amp else None,
            device=self.device
        )

        self.start_epoch = checkpoint.get("epoch", 0)
        self.best_dice = checkpoint.get("best_dice", checkpoint.get("val_dice", -1.0))
        self.patience_counter = checkpoint.get("patience_counter", 0)

        if "history" in checkpoint and isinstance(checkpoint["history"], dict):
            self.history = checkpoint["history"]

        print(
            f"[RESUME] Resumed at epoch {self.start_epoch} | Best Val Dice: {self.best_dice:.4f} | "
            f"Patience counter: {self.patience_counter}/{self.early_stopping_patience}"
        )

    def train_epoch(self, epoch: int) -> float:
        """
        Executes one complete training epoch with sample-weighted loss calculation,
        AMP autocast forward pass, and gradient clipping.
        """
        self.model.train()
        running_loss = 0.0
        total_samples = 0
        total_batches = len(self.train_loader)

        for batch_idx, batch in enumerate(self.train_loader):
            images, masks = batch[0], batch[1]
            images = images.to(self.device, non_blocking=self.is_cuda)
            masks = masks.to(self.device, non_blocking=self.is_cuda)
            batch_size = images.size(0)

            self.optimizer.zero_grad(set_to_none=True)

            # AMP forward pass
            with torch.amp.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.use_amp):
                logits = self.model(images)
                loss = self.criterion(logits, masks)

            # Backward pass with GradScaler
            self.scaler.scale(loss).backward()

            # Configurable gradient clipping with unscaling
            if self.grad_clip is not None and self.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            running_loss += loss.item() * batch_size
            total_samples += batch_size

            if (batch_idx + 1) % max(1, total_batches // 4) == 0 or (batch_idx + 1) == total_batches:
                print(f"Epoch [{epoch+1}/{self.num_epochs}] Step [{batch_idx+1}/{total_batches}] Batch Loss: {loss.item():.4f}")

        return running_loss / max(1, total_samples)

    @torch.no_grad()
    def validate_epoch(self) -> Dict[str, float]:
        """
        Executes validation in evaluation mode without gradients.
        Calculates sample-weighted validation loss and collects segmentation metrics.
        """
        self.model.eval()
        running_loss = 0.0
        total_samples = 0
        metrics_tracker = SegmentationMetricsTracker(threshold=0.5, from_logits=True)

        for batch in self.val_loader:
            images, masks = batch[0], batch[1]
            images = images.to(self.device, non_blocking=self.is_cuda)
            masks = masks.to(self.device, non_blocking=self.is_cuda)
            batch_size = images.size(0)

            with torch.amp.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.use_amp):
                logits = self.model(images)
                loss = self.criterion(logits, masks)

            running_loss += loss.item() * batch_size
            total_samples += batch_size

            # Tracker explicitly handles raw logits via sigmoid
            metrics_tracker.update(logits, masks, from_logits=True)

        val_metrics = metrics_tracker.compute()
        val_metrics["val_loss"] = float(running_loss / max(1, total_samples))
        return val_metrics

    def fit(self) -> Dict[str, Any]:
        """
        Main optimization loop. Manages scheduling, early stopping, checkpointing,
        and restoration of the best checkpoint into self.model.
        """
        print(f"\n================ STARTING TRAINING: {self.model_name.upper()} ================")
        print(
            f"Device: {self.device} | AMP: {self.use_amp} | GradClip: {self.grad_clip} | "
            f"Epochs: {self.start_epoch} -> {self.num_epochs} | "
            f"Train Batches: {len(self.train_loader)} | Val Batches: {len(self.val_loader)}"
        )
        start_time = time.time()

        for epoch in range(self.start_epoch, self.num_epochs):
            t0 = time.time()
            train_loss = self.train_epoch(epoch)
            val_metrics = self.validate_epoch()

            current_lr = self.optimizer.param_groups[0]["lr"]

            # Learning rate scheduling
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics["val_loss"])
                else:
                    self.scheduler.step()

            # Record metrics in history (full-precision)
            self.history["train_loss"].append(float(train_loss))
            self.history["val_loss"].append(float(val_metrics["val_loss"]))
            self.history["val_iou"].append(float(val_metrics["mean_iou"]))
            self.history["val_dice"].append(float(val_metrics["mean_dice"]))
            self.history["val_precision"].append(float(val_metrics["mean_precision"]))
            self.history["val_recall"].append(float(val_metrics["mean_recall"]))
            self.history["learning_rate"].append(float(current_lr))
            if "param_group_lrs" not in self.history:
                self.history["param_group_lrs"] = []
            self.history["param_group_lrs"].append([float(pg["lr"]) for pg in self.optimizer.param_groups])

            epoch_time = time.time() - t0
            if len(self.optimizer.param_groups) > 1:
                lr_display = " | ".join([f"LR_g{i}: {pg['lr']:.6f}" for i, pg in enumerate(self.optimizer.param_groups)])
            else:
                lr_display = f"LR: {current_lr:.6f}"
            print(
                f"[Epoch {epoch+1}/{self.num_epochs} - {epoch_time:.1f}s | {lr_display}] "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_metrics['val_loss']:.4f} | "
                f"Val Dice: {val_metrics['mean_dice']:.4f} | Val IoU: {val_metrics['mean_iou']:.4f}"
            )

            # Early stopping and best model tracking
            is_best = val_metrics["mean_dice"] > (self.best_dice + self.min_delta)
            if is_best:
                self.best_dice = val_metrics["mean_dice"]
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            # Prepare comprehensive state payload
            state = {
                "epoch": epoch + 1,
                "model_name": self.model_name,
                "state_dict": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
                "scaler": self.scaler.state_dict() if (self.scaler is not None and self.use_amp) else None,
                "best_dice": self.best_dice,
                "patience_counter": self.patience_counter,
                "history": self.history,
                "val_dice": val_metrics["mean_dice"],
                "val_iou": val_metrics["mean_iou"],
                "val_loss": val_metrics["val_loss"]
            }
            save_checkpoint(state, is_best, checkpoint_dir=self.checkpoint_dir, model_name=self.model_name)

            if self.patience_counter >= self.early_stopping_patience:
                print(f"[EARLY STOPPING] Patience of {self.early_stopping_patience} epochs reached at epoch {epoch+1}.")
                break

        total_duration = time.time() - start_time
        print(f"\nTraining completed in {total_duration:.1f}s. Best Val Dice: {self.best_dice:.4f}")

        # Correctly restore the best checkpoint into self.model
        self.restore_best_model()

        # Save logs and curves
        self._save_training_artifacts(total_duration)
        return self.history

    def restore_best_model(self) -> bool:
        """
        Restores in-memory self.model weights from the saved best checkpoint.
        Ensures that downstream inference or evaluation uses optimal weights.
        """
        best_path = Path(self.checkpoint_dir) / f"{self.model_name}_best.pth"
        if best_path.exists():
            checkpoint = torch.load(best_path, map_location=self.device)
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["state_dict"])
            elif isinstance(checkpoint, dict):
                self.model.load_state_dict(checkpoint)
            best_ep = checkpoint.get("epoch", "unknown") if isinstance(checkpoint, dict) else "unknown"
            print(f"[TRAINER] Restored in-memory model to best checkpoint from epoch {best_ep} (Best Val Dice: {self.best_dice:.4f})")
            return True
        else:
            print(f"[TRAINER WARNING] Best checkpoint not found at {best_path}; model weights remain at final epoch.")
            return False

    def _save_training_artifacts(self, duration: float):
        """
        Saves full JSON metrics history and plots 2-panel loss/metric curve diagrams.
        """
        os.makedirs(self.reports_dir, exist_ok=True)
        history_path = Path(self.reports_dir) / f"{self.model_name}_training_history.json"
        summary = {
            "model_name": self.model_name,
            "training_duration_seconds": round(duration, 2),
            "best_val_dice": round(float(self.best_dice), 4),
            "final_metrics": {
                k: ([round(float(x), 4) for x in v[-1]] if isinstance(v[-1], list) else round(float(v[-1]), 4))
                for k, v in self.history.items() if v
            },
            "history": self.history
        }
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        # Plot training curves
        num_recorded = len(self.history["train_loss"])
        if num_recorded > 0:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            epochs_axis = range(1, num_recorded + 1)

            axes[0].plot(epochs_axis, self.history["train_loss"], label="Train Loss", color="#d9534f")
            axes[0].plot(epochs_axis, self.history["val_loss"], label="Val Loss", color="#0275d8")
            axes[0].set_title(f"{self.model_name.upper()} - Loss Curves")
            axes[0].set_xlabel("Epoch")
            axes[0].set_ylabel("Loss")
            axes[0].legend()
            axes[0].grid(True, linestyle="--", alpha=0.6)

            axes[1].plot(epochs_axis, self.history["val_dice"], label="Val Dice", color="#5cb85c")
            axes[1].plot(epochs_axis, self.history["val_iou"], label="Val IoU", color="#f0ad4e")
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
