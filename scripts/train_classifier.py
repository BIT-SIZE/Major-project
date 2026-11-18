import os
import sys
import time
import json
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from ml.models.classifier import PlantDiseaseClassifier
from ml.dataset.classification_loader import get_classification_dataloaders

def parse_args():
    parser = argparse.ArgumentParser(description="Train EfficientNet-B0 Disease Classifier on PlantSeg")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--checkpoint-dir", type=str, default="models/checkpoints")
    parser.add_argument("--reports-dir", type=str, default="reports/training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic class-aware sampling")
    return parser.parse_args()

def calculate_topk_accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1, 3)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return res

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print(f"TRAINING EFFICIENTNET-B0 PLANT DISEASE CLASSIFIER")
    print(f"Device: {device} | Epochs: {args.epochs} | Batch Size: {args.batch_size}")
    print("=" * 60)

    # 1. Load Data
    data = get_classification_dataloaders(
        root_dir=str(ROOT_DIR),
        image_size=args.image_size,
        batch_size=args.batch_size,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        seed=args.seed
    )
    train_loader = data["train"]
    val_loader = data["val"]
    mapping = data["mapping"]
    num_classes = mapping["num_classes"]

    print(f"[DATA] Train Batches: {len(train_loader)} | Val Batches: {len(val_loader)} | Classes: {num_classes}")

    # 2. Build Model
    model = PlantDiseaseClassifier(num_classes=num_classes, pretrained=True)
    model.to(device)

    # 3. Loss, Optimizer, Scheduler
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    best_val_acc = -1.0
    history = {
        "train_loss": [],
        "train_acc_top1": [],
        "val_loss": [],
        "val_acc_top1": [],
        "val_acc_top3": [],
        "learning_rate": []
    }

    ckpt_dir = ROOT_DIR / args.checkpoint_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = ROOT_DIR / args.reports_dir
    reports_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    for epoch in range(args.epochs):
        t0 = time.time()
        model.train()
        running_loss = 0.0
        train_top1_accs = []
        total_train_steps = len(train_loader)
        current_lr = optimizer.param_groups[0]["lr"]

        for step, (images, targets, _) in enumerate(train_loader):
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            acc1, _ = calculate_topk_accuracy(outputs, targets, topk=(1, 3))
            train_top1_accs.append(acc1)

            if (step + 1) % 25 == 0 or (step + 1) == total_train_steps:
                print(f"Epoch [{epoch+1}/{args.epochs}] Step [{step+1}/{total_train_steps}] Loss: {loss.item():.4f} | Batch Acc: {acc1:.1f}%", flush=True)

        train_loss = running_loss / max(1, total_train_steps)
        mean_train_top1 = float(np.mean(train_top1_accs)) if train_top1_accs else 0.0

        # Validation Loop
        model.eval()
        val_loss = 0.0
        top1_accs = []
        top3_accs = []

        with torch.no_grad():
            for images, targets, _ in val_loader:
                images = images.to(device)
                targets = targets.to(device)

                outputs = model(images)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

                acc1, acc3 = calculate_topk_accuracy(outputs, targets, topk=(1, 3))
                top1_accs.append(acc1)
                top3_accs.append(acc3)

        val_loss = val_loss / max(1, len(val_loader))
        mean_top1 = float(np.mean(top1_accs)) if top1_accs else 0.0
        mean_top3 = float(np.mean(top3_accs)) if top3_accs else 0.0

        scheduler.step()

        history["train_loss"].append(round(train_loss, 4))
        history["train_acc_top1"].append(round(mean_train_top1, 2))
        history["val_loss"].append(round(val_loss, 4))
        history["val_acc_top1"].append(round(mean_top1, 2))
        history["val_acc_top3"].append(round(mean_top3, 2))
        history["learning_rate"].append(current_lr)

        elapsed = time.time() - t0
        print(f"[Epoch {epoch+1}/{args.epochs} - {elapsed:.1f}s | LR: {current_lr:.6f}] Train Loss: {train_loss:.4f} (Acc: {mean_train_top1:.2f}%) | Val Loss: {val_loss:.4f} | Top-1: {mean_top1:.2f}% | Top-3: {mean_top3:.2f}%")

        # Checkpoint saving
        is_best = mean_top1 > best_val_acc
        if is_best:
            best_val_acc = mean_top1
            save_payload = {
                "epoch": epoch + 1,
                "model_name": "EfficientNet-B0",
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "val_acc_top1": mean_top1,
                "val_acc_top3": mean_top3,
                "val_loss": val_loss,
                "mapping": mapping
            }
            best_ckpt = ckpt_dir / "efficientnet_b0_best.pth"
            torch.save(save_payload, best_ckpt)
            print(f"[CHECKPOINT] Saved new best classifier to {best_ckpt} (Top-1: {mean_top1:.2f}%)")

    total_duration = time.time() - start_time
    print(f"\n[COMPLETE] Classification training finished in {total_duration:.1f}s. Best Val Top-1: {best_val_acc:.2f}%")

    # Save final epoch checkpoint
    last_payload = {
        "epoch": args.epochs,
        "model_name": "EfficientNet-B0",
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "val_acc_top1": mean_top1,
        "val_acc_top3": mean_top3,
        "val_loss": val_loss,
        "mapping": mapping
    }
    last_ckpt = ckpt_dir / "efficientnet_b0_last.pth"
    torch.save(last_payload, last_ckpt)
    print(f"[CHECKPOINT] Saved final epoch classifier to {last_ckpt}")

    # Save training configuration
    config_file = ckpt_dir / "training_config.json"
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump({
            "model_architecture": "EfficientNet-B0",
            "pretrained": True,
            "num_classes": num_classes,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "weight_decay": 1e-4,
            "label_smoothing": 0.05,
            "scheduler": "CosineAnnealingLR",
            "image_size": args.image_size,
            "seed": args.seed,
            "max_train_samples": args.max_train_samples,
            "max_val_samples": args.max_val_samples,
            "checkpoint_dir": str(args.checkpoint_dir)
        }, f, indent=2)

    # Save class mapping reference
    with open(ckpt_dir / "disease_classes.json", "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2)

    # Save training logs and curves
    history_payload = {
        "model": "EfficientNet-B0",
        "epochs": args.epochs,
        "best_val_acc_top1": best_val_acc,
        "duration_seconds": round(total_duration, 2),
        "history": history
    }
    history_file = reports_dir / "classifier_training_history.json"
    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(history_payload, f, indent=2)

    # Also save copy in ckpt_dir
    with open(ckpt_dir / "classifier_training_history.json", "w", encoding="utf-8") as f:
        json.dump(history_payload, f, indent=2)

    # Plot curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ep_range = range(1, len(history["train_loss"]) + 1)

    axes[0].plot(ep_range, history["train_loss"], label="Train Loss", color="#d9534f")
    axes[0].plot(ep_range, history["val_loss"], label="Val Loss", color="#0275d8")
    axes[0].set_title("EfficientNet-B0 - Cross Entropy Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.6)

    axes[1].plot(ep_range, history["train_acc_top1"], label="Train Top-1 Acc (%)", color="#0275d8", linestyle="--")
    axes[1].plot(ep_range, history["val_acc_top1"], label="Val Top-1 Acc (%)", color="#5cb85c")
    axes[1].plot(ep_range, history["val_acc_top3"], label="Val Top-3 Acc (%)", color="#f0ad4e")
    axes[1].set_title("Accuracy Curves (Train & Val)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    curve_fig = reports_dir / "classifier_training_curves.png"
    plt.savefig(curve_fig, dpi=120)
    plt.savefig(ckpt_dir / "classifier_training_curves.png", dpi=120)
    plt.close()
    print(f"[REPORTS] Saved curves figure to {curve_fig} and {ckpt_dir / 'classifier_training_curves.png'}")

if __name__ == "__main__":
    main()
