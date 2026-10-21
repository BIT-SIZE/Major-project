import json
import pytest
import numpy as np
import torch
from pathlib import Path

from ml.models.classifier import PlantDiseaseClassifier
from backend.app.services.classification_service import ClassificationService

def test_class_mapping_integrity():
    mapping_path = Path("models/checkpoints/disease_classes.json")
    assert mapping_path.exists(), "disease_classes.json must exist"
    with open(mapping_path, "r", encoding="utf-8") as f:
        mapping = json.load(f)

    assert mapping["num_classes"] == 114
    assert len(mapping["classes"]) == 114
    assert len(mapping["disease_to_idx"]) == 114
    assert len(mapping["disease_to_plant"]) == 114

def test_classifier_forward_and_topk():
    model = PlantDiseaseClassifier(num_classes=114, pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    logits = model(x)
    assert logits.shape == (2, 114)

    scores, indices = model.predict_top_k(x, k=3)
    assert scores.shape == (2, 3)
    assert indices.shape == (2, 3)
    # Probability sum <= 1.0
    assert torch.all(scores >= 0.0)
    assert torch.all(scores <= 1.0)

def test_classification_service_predict():
    service = ClassificationService()
    assert service.is_available() is True

    # Generate synthetic leaf RGB image
    img = np.zeros((224, 224, 3), dtype=np.uint8)
    img[:, :] = [34, 139, 34]  # Forest green

    result = service.classify(img, top_k=3)
    assert result["status"] == "success"
    assert result["predicted_disease"] is not None
    assert result["host_plant"] is not None
    assert len(result["top_predictions"]) == 3
    assert result["confidence_score"] > 0.0

def test_low_confidence_uncertainty():
    # If confidence threshold is set high (e.g. 0.99), synthetic image should trigger uncertainty
    service = ClassificationService(confidence_threshold=0.99)
    img = np.zeros((224, 224, 3), dtype=np.uint8)
    result = service.classify(img, top_k=3)
    assert result["uncertainty_warning"] is True
    assert "Confidence is low" in result["message"]
