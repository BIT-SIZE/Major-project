import os
import sys
import argparse
from pathlib import Path
import yaml
import torch
import torch.optim as optim

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from ml.models.unet import UNet
from ml.dataset.dataset_loader import get_dataloaders
from ml.training.losses import get_loss_function
from ml.training.trainer import SegmentationTrainer

def parse_args():
    parser = argparse.ArgumentParser(description="Train U-Net on PlantSeg dataset")
    parser.add_argument("--config", type=str, default="ml/configs/unet.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training from")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=256)
    return parser.parse_args()

def main():
    args = parse_args()
    config_path = ROOT_DIR / args.config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    epochs = args.epochs or config["training"]["epochs"]
    batch_size = args.batch_size or config["dataset"]["batch_size"]
    lr = args.lr or config["training"]["learning_rate"]
    img_size = (args.image_size, args.image_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[SETUP] Starting U-Net Training on device: {device}")

    # Build model
    model = UNet(
        n_channels=3,
        n_classes=config["model"]["num_classes"],
        base_features=config["model"].get("base_features", 32)
    )

    # Get DataLoaders
    loaders = get_dataloaders(
        root_dir=str(ROOT_DIR),
        image_size=img_size,
        batch_size=batch_size,
        num_workers=config["dataset"].get("num_workers", 2),
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples
    )

    # Loss, Optimizer & Scheduler
    criterion = get_loss_function(config["training"].get("loss", "bce_dice"))
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=config["training"].get("weight_decay", 1e-4))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    trainer = SegmentationTrainer(
        model=model,
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=epochs,
        model_name="unet",
        checkpoint_dir=str(ROOT_DIR / config["training"]["checkpoint_dir"]),
        reports_dir=str(ROOT_DIR / config["training"]["reports_dir"]),
        early_stopping_patience=config["training"].get("early_stopping_patience", 6),
        use_amp=config["training"].get("use_amp", True),
        grad_clip=config["training"].get("grad_clip", 1.0),
        resume_path=args.resume
    )

    history = trainer.fit()
    print("[SUCCESS] U-Net training pipeline completed successfully.")

if __name__ == "__main__":
    main()
