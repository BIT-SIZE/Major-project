import io
import pytest
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

from backend.app.main import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"

def test_health():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "has_trained_model" in data
    assert data["has_trained_model"] is True
    assert "selected_classification_model" in data

def test_get_models():
    response = client.get("/api/models")
    assert response.status_code == 200
    data = response.json()
    assert len(data["available_models"]) >= 4
    model_names = [m["name"] for m in data["available_models"]]
    assert "EfficientNet-B0" in model_names
    assert "DeepLabV3+" in model_names
    assert "U-Net" in model_names
    assert "SegFormer-B0" in model_names

def test_models_comparison():
    response = client.get("/api/models/comparison")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "available"
    assert "segmentation_benchmarks" in data
    assert "classification_benchmarks" in data

def test_analytics_summary():
    # Protected endpoint requires authentication
    unauth = client.get("/api/analytics/summary")
    assert unauth.status_code == 401

    response = client.get("/api/analytics/summary", headers={"Authorization": "Bearer test-mock-token"})
    assert response.status_code == 200
    data = response.json()
    assert "total_predictions" in data
    assert "average_severity_percentage" in data

def test_dataset_summary():
    response = client.get("/api/dataset/summary")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "inspected"
    assert data["data"]["total_metadata_records"] == 11458

def test_standalone_classify():
    img = Image.new("RGB", (224, 224), color=(34, 139, 34))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    response = client.post(
        "/api/classify",
        files={"file": ("leaf.jpg", buf, "image/jpeg")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["predicted_disease"] is not None
    assert data["host_plant"] is not None
    assert len(data["top_predictions"]) >= 1
    assert data["model_name"] == "EfficientNet-B0"

from unittest.mock import patch
from backend.app.core.config import settings

@patch.object(settings, "DATABASE_BACKEND", "sqlite")
def test_predict_and_history_lifecycle():
    # 0. Verify unauthenticated call is rejected
    img = Image.new("RGB", (100, 100), color=(34, 139, 34))
    buf_unauth = io.BytesIO()
    img.save(buf_unauth, format="JPEG")
    buf_unauth.seek(0)
    unauth_resp = client.post(
        "/api/predict",
        files={"file": ("test_leaf.jpg", buf_unauth, "image/jpeg")}
    )
    assert unauth_resp.status_code == 401

    # 1. Post prediction with valid auth header
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    auth_headers = {"Authorization": "Bearer test-mock-token"}

    response = client.post(
        "/api/predict",
        files={"file": ("test_leaf.jpg", buf, "image/jpeg")},
        data={"model_name": "deeplabv3plus"},
        headers=auth_headers
    )
    assert response.status_code == 200
    pred = response.json()
    assert pred["status"] == "success"
    assert "prediction_id" in pred
    pred_id = pred["prediction_id"]

    # Verify complementary blocks
    assert "classification" in pred
    assert pred["classification"]["status"] == "success"
    assert pred["classification"]["predicted_disease"] is not None
    assert pred["classification"]["host_plant"] is not None
    assert len(pred["classification"]["top_predictions"]) >= 1

    assert "segmentation" in pred
    assert pred["segmentation"]["disease_segmentation_available"] is True

    assert "severity" in pred
    assert pred["leaf_area_available"] is True
    assert pred["overlay_image_url"] is not None

    assert "inference" in pred
    assert pred["inference"]["total_time_ms"] > 0

    # 2. Check history (scoped to authenticated user)
    h_resp = client.get("/api/history", headers=auth_headers)
    assert h_resp.status_code == 200
    h_data = h_resp.json()
    assert h_data["total"] >= 1
    ids = [item["id"] for item in h_data["items"]]
    assert pred_id in ids

    # 3. Get detail
    d_resp = client.get(f"/api/history/{pred_id}", headers=auth_headers)
    assert d_resp.status_code == 200
    detail = d_resp.json()
    assert detail["id"] == pred_id
    assert detail["predicted_disease"] is not None

    # 4. Delete
    del_resp = client.delete(f"/api/history/{pred_id}", headers=auth_headers)
    assert del_resp.status_code == 200
    assert del_resp.json()["status"] == "success"
