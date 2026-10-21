import os
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image

from ml.models.classifier import PlantDiseaseClassifier
from backend.app.core.config import settings

class ClassificationService:
    """
    Dedicated Plant Disease Classification Service.
    Loads EfficientNet-B0 checkpoint and mapping file once on startup.
    Computes top-k ranked disease predictions, host plant identification, and confidence calibration.
    """
    def __init__(
        self,
        checkpoint_path: str = "models/checkpoints/efficientnet_b0_best.pth",
        mapping_path: str = "models/checkpoints/disease_classes.json",
        confidence_threshold: float = 0.25
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.mapping_path = Path(mapping_path)
        self.confidence_threshold = confidence_threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model: Optional[PlantDiseaseClassifier] = None
        self.mapping: Optional[Dict[str, Any]] = None
        self.classes: List[str] = []
        self.disease_to_plant: Dict[str, str] = {}
        self.num_classes: int = 114

        # ImageNet normalization parameters
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        self._load_service()

    def _load_service(self):
        # 1. Load label mapping
        if self.mapping_path.exists():
            with open(self.mapping_path, "r", encoding="utf-8") as f:
                self.mapping = json.load(f)
                self.classes = self.mapping.get("classes", [])
                self.disease_to_plant = self.mapping.get("disease_to_plant", {})
                self.num_classes = len(self.classes)
                print(f"[CLASSIFICATION SERVICE] Loaded {self.num_classes} disease classes mapping.")
        else:
            print(f"[CLASSIFICATION SERVICE] Mapping file not found at {self.mapping_path}.")

        # 2. Load model checkpoint if available
        if self.checkpoint_path.exists():
            try:
                ckpt = torch.load(self.checkpoint_path, map_location=self.device)
                self.model = PlantDiseaseClassifier(num_classes=self.num_classes, pretrained=False)
                self.model.load_state_dict(ckpt["state_dict"])
                self.model.to(self.device)
                self.model.eval()
                print(f"[CLASSIFICATION SERVICE] Loaded EfficientNet-B0 classifier checkpoint on {self.device}.")
            except Exception as e:
                print(f"[CLASSIFICATION SERVICE] Failed to load checkpoint: {e}")
                self.model = None
        else:
            print(f"[CLASSIFICATION SERVICE] Checkpoint {self.checkpoint_path} not found. Running in untrained/fallback mode.")

    def is_available(self) -> bool:
        return self.model is not None and self.num_classes > 0

    def classify(self, image_rgb: np.ndarray, top_k: int = 3) -> Dict[str, Any]:
        """
        Runs disease classification on an RGB image.
        Returns top_predictions list, primary predicted_disease, host_plant, and confidence.
        """
        if not self.is_available():
            return {
                "status": "untrained_model",
                "predicted_disease": None,
                "host_plant": None,
                "confidence_score": None,
                "top_predictions": [],
                "model_name": "EfficientNet-B0",
                "inference_time_ms": 0.0,
                "uncertainty_warning": True,
                "message": "Disease classifier has not been trained or checkpoint is unavailable."
            }

        t0 = time.time()

        # Preprocessing: 224x224 RGB ImageNet normalized
        resized = cv2.resize(image_rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
        norm_img = (resized.astype(np.float32) / 255.0 - self.mean) / self.std
        tensor_img = torch.from_numpy(norm_img.transpose(2, 0, 1)).unsqueeze(0).float().to(self.device)

        with torch.no_grad():
            logits = self.model(tensor_img)
            probs = F.softmax(logits, dim=1)
            k = min(top_k, self.num_classes)
            top_scores, top_indices = torch.topk(probs, k=k, dim=1)

        latency_ms = round((time.time() - t0) * 1000.0, 2)

        scores = top_scores.squeeze().cpu().numpy().tolist()
        indices = top_indices.squeeze().cpu().numpy().tolist()

        if not isinstance(scores, list):
            scores = [scores]
            indices = [indices]

        top_predictions = []
        for rank, (idx, score) in enumerate(zip(indices, scores)):
            disease_name = self.classes[idx] if idx < len(self.classes) else f"Class #{idx}"
            host_plant = self.disease_to_plant.get(disease_name, "Unknown")
            top_predictions.append({
                "rank": rank + 1,
                "disease": disease_name.title(),
                "host_plant": host_plant,
                "score": round(float(score), 4),
                "confidence_pct": round(float(score) * 100.0, 1)
            })

        primary_pred = top_predictions[0] if top_predictions else None
        top_score = primary_pred["score"] if primary_pred else 0.0

        is_uncertain = top_score < self.confidence_threshold

        return {
            "status": "success",
            "predicted_disease": primary_pred["disease"] if primary_pred else None,
            "host_plant": primary_pred["host_plant"] if primary_pred else None,
            "confidence_score": top_score,
            "confidence_pct": primary_pred["confidence_pct"] if primary_pred else None,
            "top_predictions": top_predictions,
            "model_name": "EfficientNet-B0",
            "inference_time_ms": latency_ms,
            "uncertainty_warning": is_uncertain,
            "message": (
                "Classification successful." if not is_uncertain else
                f"Confidence is low ({primary_pred['confidence_pct']}%). Foliage symptoms may be ambiguous or outside supported categories."
            )
        }
