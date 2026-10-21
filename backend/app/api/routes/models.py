import os
import json
from pathlib import Path
from fastapi import APIRouter
from pydantic import BaseModel
import torch

from backend.app.schemas.predictions import ModelsListResponse

router = APIRouter()

@router.get("/models", response_model=ModelsListResponse)
def get_models():
    ckpt_dir = Path("models/checkpoints")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    models_info = [
        {
            "name": "EfficientNet-B0",
            "architecture": "efficientnet_b0",
            "task_type": "classification",
            "has_checkpoint": (ckpt_dir / "efficientnet_b0_best.pth").exists(),
            "is_active": True,
            "parameters": "4,153,582",
            "checkpoint_size_mb": round(os.path.getsize(ckpt_dir / "efficientnet_b0_best.pth") / (1024 * 1024), 2) if (ckpt_dir / "efficientnet_b0_best.pth").exists() else None,
            "accuracy_top1": 73.33,
            "accuracy_top3": 100.0,
            "avg_latency_ms": 8.85,
            "description": "Pretrained EfficientNet-B0 fine-tuned on 114 PlantSeg disease categories. Computes Top-3 ranked diagnoses with calibrated confidence scores."
        },
        {
            "name": "DeepLabV3+",
            "architecture": "deeplabv3plus",
            "task_type": "segmentation",
            "has_checkpoint": (ckpt_dir / "deeplabv3plus_best.pth").exists(),
            "is_active": True,
            "parameters": "40,347,041",
            "checkpoint_size_mb": round(os.path.getsize(ckpt_dir / "deeplabv3plus_best.pth") / (1024 * 1024), 2) if (ckpt_dir / "deeplabv3plus_best.pth").exists() else None,
            "dice_coefficient": 0.5483,
            "mean_iou": 0.4403,
            "avg_latency_ms": 81.1,
            "description": "Atrous Spatial Pyramid Pooling (ASPP) with ResNet-50 backbone for multi-scale plant lesion segmentation."
        },
        {
            "name": "U-Net",
            "architecture": "unet",
            "task_type": "segmentation",
            "has_checkpoint": (ckpt_dir / "unet_best.pth").exists(),
            "is_active": False,
            "parameters": "4,318,401",
            "checkpoint_size_mb": round(os.path.getsize(ckpt_dir / "unet_best.pth") / (1024 * 1024), 2) if (ckpt_dir / "unet_best.pth").exists() else None,
            "dice_coefficient": 0.3719,
            "mean_iou": 0.2668,
            "avg_latency_ms": 76.14,
            "description": "Encoder-decoder baseline network with skip connections preserving fine spatial lesion details."
        },
        {
            "name": "SegFormer-B0",
            "architecture": "segformer",
            "task_type": "segmentation",
            "has_checkpoint": (ckpt_dir / "segformer_best.pth").exists(),
            "is_active": False,
            "parameters": "3,711,329",
            "checkpoint_size_mb": round(os.path.getsize(ckpt_dir / "segformer_best.pth") / (1024 * 1024), 2) if (ckpt_dir / "segformer_best.pth").exists() else None,
            "dice_coefficient": 0.2964,
            "mean_iou": 0.2020,
            "avg_latency_ms": 18.37,
            "description": "Hierarchical Mix-Transformer (MiT-B0) encoder with lightweight All-MLP decoder for ultra-fast attention-based segmentation."
        }
    ]

    active_seg = "deeplabv3plus" if (ckpt_dir / "deeplabv3plus_best.pth").exists() else ("unet" if (ckpt_dir / "unet_best.pth").exists() else "none")

    return {
        "active_segmentation_model": active_seg,
        "classification_model": "EfficientNet-B0",
        "device": device,
        "available_models": models_info
    }


@router.get("/models/comparison")
def get_model_comparison():
    seg_report_path = Path("reports/evaluation/model_comparison_report.json")
    clf_report_path = Path("reports/evaluation/classifier_evaluation_report.json")

    seg_data = []
    if seg_report_path.exists():
        with open(seg_report_path, "r", encoding="utf-8") as f:
            seg_data = json.load(f).get("benchmark_results", [])

    clf_data = {}
    if clf_report_path.exists():
        with open(clf_report_path, "r", encoding="utf-8") as f:
            clf_data = json.load(f)

    return {
        "status": "available",
        "benchmark_results": seg_data,
        "segmentation_benchmarks": seg_data,
        "classification_benchmarks": clf_data
    }
