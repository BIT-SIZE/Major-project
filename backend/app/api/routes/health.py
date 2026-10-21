from pathlib import Path
from fastapi import APIRouter
import torch

router = APIRouter()

@router.get("/health")
def get_health():
    checkpoints_dir = Path("models/checkpoints")
    models_available = {
        "unet": (checkpoints_dir / "unet_best.pth").exists(),
        "deeplabv3plus": (checkpoints_dir / "deeplabv3plus_best.pth").exists(),
        "segformer": (checkpoints_dir / "segformer_best.pth").exists(),
        "efficientnet_b0": (checkpoints_dir / "efficientnet_b0_best.pth").exists()
    }

    has_segmentation = any([models_available["unet"], models_available["deeplabv3plus"], models_available["segformer"]])
    has_classifier = models_available["efficientnet_b0"]
    selected_seg = "deeplabv3plus" if models_available["deeplabv3plus"] else ("unet" if models_available["unet"] else "none")

    return {
        "status": "healthy",
        "service": "Food Security Intelligent Crop Health Monitoring API",
        "version": "1.1.0",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "cuda_available": torch.cuda.is_available(),
        "has_trained_model": has_segmentation and has_classifier,
        "selected_segmentation_model": selected_seg,
        "selected_classification_model": "efficientnet_b0" if has_classifier else "none",
        "model_checkpoints": models_available
    }
