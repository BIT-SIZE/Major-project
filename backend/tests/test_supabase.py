import os
import json
import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone

from backend.app.core.config import settings
from backend.app.services.supabase_storage import SupabaseStorageService
from backend.app.db.repository import PredictionRepository, SupabaseRecordAdapter
from backend.app.db.supabase_client import get_supabase_client, reset_supabase_client

@pytest.fixture(autouse=True)
def reset_client():
    reset_supabase_client()
    yield
    reset_supabase_client()

def test_supabase_config_detection():
    """Verify is_supabase_configured correctly validates presence of non-placeholder credentials."""
    assert not settings.is_supabase_configured or settings.SUPABASE_URL is not None

def test_supabase_storage_local_fallback_when_unconfigured():
    """When credentials are not set, SupabaseStorageService gracefully reports unconfigured."""
    with patch.object(settings, "SUPABASE_URL", None), patch.object(settings, "SUPABASE_SERVICE_ROLE_KEY", None):
        reset_supabase_client()
        storage = SupabaseStorageService()
        assert not storage.is_available
        res = storage.upload_file("crop-images", "test.jpg", b"fakebytes")
        assert not res["success"]
        assert res["signed_url"] is None

def test_supabase_storage_upload_and_signed_url_mocked():
    """Verify SupabaseStorageService uploads and creates signed URLs using Supabase SDK."""
    mock_client = MagicMock()
    mock_bucket = MagicMock()
    mock_client.storage.from_.return_value = mock_bucket
    mock_bucket.create_signed_url.return_value = {"signedURL": "https://test.supabase.co/signed/test.jpg"}

    with patch("backend.app.services.supabase_storage.get_supabase_client", return_value=mock_client):
        storage = SupabaseStorageService()
        assert storage.is_available

        upload_res = storage.upload_file("crop-images", "predictions/123_orig.jpg", b"test_bytes")
        assert upload_res["success"] is True
        assert upload_res["storage_path"] == "predictions/123_orig.jpg"
        assert upload_res["signed_url"] == "https://test.supabase.co/signed/test.jpg"

        mock_bucket.upload.assert_called_once()
        mock_bucket.create_signed_url.assert_called_once()

def test_supabase_storage_error_handling():
    """Verify storage handles network and API exceptions gracefully without crashing."""
    mock_client = MagicMock()
    mock_bucket = MagicMock()
    mock_bucket.upload.side_effect = Exception("Storage quota exceeded")
    mock_client.storage.from_.return_value = mock_bucket

    with patch("backend.app.services.supabase_storage.get_supabase_client", return_value=mock_client):
        storage = SupabaseStorageService()
        upload_res = storage.upload_file("crop-images", "predictions/error.jpg", b"bytes")
        assert upload_res["success"] is False
        assert "Storage quota exceeded" in upload_res["error"]

def test_supabase_record_adapter():
    """Verify SupabaseRecordAdapter maps columns, handles ISO timestamps, and provides fallback attributes."""
    raw_row = {
        "id": "pred-uuid-1",
        "created_at": "2026-10-27T03:00:00Z",
        "predicted_disease": "apple mosaic virus",
        "predicted_host": "apple",
        "classification_score": 0.94,
        "model_name": "DeepLabV3+",
        "severity_percentage": 12.5,
        "severity_category": "Moderate",
        "image_path": "predictions/pred-uuid-1_orig.jpg",
        "mask_path": "masks/pred-uuid-1_mask.png",
        "overlay_path": "overlays/pred-uuid-1_overlay.jpg",
        "top_predictions": [
            {"disease": "apple mosaic virus", "score": 0.94},
            {"disease": "apple rust", "score": 0.04}
        ]
    }

    mock_storage = MagicMock()
    mock_storage.is_available = True
    mock_storage.bucket_images = "crop-images"
    mock_storage.bucket_masks = "segmentation-masks"
    mock_storage.create_signed_url.side_effect = lambda b, p: f"https://cdn.supabase.co/{b}/{p}"

    adapter = SupabaseRecordAdapter(raw_row, mock_storage)
    assert adapter.id == "pred-uuid-1"
    assert adapter.predicted_disease == "apple mosaic virus"
    assert adapter.predicted_host == "apple"
    assert adapter.severity_percentage == 12.5
    assert adapter.severity_category == "Moderate"
    assert adapter.original_image_path == "https://cdn.supabase.co/crop-images/predictions/pred-uuid-1_orig.jpg"
    assert adapter.top_predictions is not None
    assert len(adapter.top_predictions) == 2
    assert adapter.top_predictions_json is not None

def test_prediction_repository_supabase_insert_and_model_runs():
    """Verify repository inserts into both predictions and model_runs tables when Supabase is active."""
    mock_client = MagicMock()
    mock_pred_table = MagicMock()
    mock_model_runs_table = MagicMock()

    def table_router(table_name):
        if table_name == "predictions":
            return mock_pred_table
        if table_name == "model_runs":
            return mock_model_runs_table
        return MagicMock()

    mock_client.table.side_effect = table_router
    mock_pred_table.insert.return_value.execute.return_value.data = [{
        "id": "pred-123",
        "created_at": "2026-10-27T03:00:00Z",
        "predicted_disease": "tomato early blight",
        "predicted_host": "tomato",
        "model_name": "DeepLabV3+",
        "severity_percentage": 5.4,
        "severity_category": "Mild",
        "image_path": "predictions/pred-123_orig.jpg",
        "mask_path": "masks/pred-123_mask.png"
    }]

    with patch("backend.app.db.repository.get_supabase_client", return_value=mock_client), \
         patch.object(settings, "DATABASE_BACKEND", "supabase"):
        repo = PredictionRepository(db=None)
        assert repo.use_supabase is True

        created = repo.create({
            "id": "pred-123",
            "original_filename": "leaf.jpg",
            "image_path": "predictions/pred-123_orig.jpg",
            "mask_path": "masks/pred-123_mask.png",
            "overlay_path": "overlays/pred-123_overlay.jpg",
            "predicted_disease": "tomato early blight",
            "predicted_host": "tomato",
            "classification_score": 0.88,
            "classification_time_ms": 10.5,
            "inference_time_ms": 85.0,
            "severity_percentage": 5.4,
            "severity_category": "Mild",
            "model_name": "DeepLabV3+"
        })

        assert created.id == "pred-123"
        assert created.predicted_disease == "tomato early blight"
        mock_pred_table.insert.assert_called_once()
        mock_model_runs_table.insert.assert_called_once()

def test_prediction_repository_supabase_query_methods():
    """Verify get_all, get_by_id, count, and delete methods through Supabase client."""
    mock_client = MagicMock()
    mock_table = MagicMock()
    mock_client.table.return_value = mock_table

    mock_table.select.return_value.order.return_value.range.return_value.execute.return_value.data = [
        {"id": "p1", "predicted_disease": "apple scab", "created_at": "2026-10-27T00:00:00Z"}
    ]
    mock_table.select.return_value.execute.return_value.count = 1

    with patch("backend.app.db.repository.get_supabase_client", return_value=mock_client), \
         patch.object(settings, "DATABASE_BACKEND", "supabase"):
        repo = PredictionRepository(db=None)

        items = repo.get_all(skip=0, limit=10)
        assert len(items) == 1
        assert items[0].id == "p1"

        cnt = repo.count()
        assert cnt == 1

        deleted = repo.delete("p1")
        assert deleted is True
        mock_table.delete.assert_called()
