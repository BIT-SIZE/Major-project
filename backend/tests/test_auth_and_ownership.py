import io
import uuid
import pytest
from unittest.mock import MagicMock, patch
from PIL import Image
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.core.security import AuthenticatedUser, get_current_user
from backend.app.db.repository import PredictionRepository

client = TestClient(app)

USER_A_ID = "11111111-1111-1111-1111-111111111111"
USER_B_ID = "22222222-2222-2222-2222-222222222222"

user_a = AuthenticatedUser(id=USER_A_ID, email="user_a@test.dev", is_anonymous=False)
user_b = AuthenticatedUser(id=USER_B_ID, email="user_b@test.dev", is_anonymous=True)

def _create_test_image():
    img = Image.new("RGB", (64, 64), color=(34, 139, 34))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf

# 1. Token Validation Tests
def test_missing_auth_token_rejected():
    res = client.get("/api/history")
    assert res.status_code == 401
    assert "Authentication credentials were not provided" in res.json()["detail"]

def test_malformed_auth_header_rejected():
    res = client.get("/api/history", headers={"Authorization": "Basic invalid-credentials"})
    assert res.status_code == 401
    assert "Invalid authorization header format" in res.json()["detail"]

    res2 = client.get("/api/history", headers={"Authorization": "Bearer "})
    assert res2.status_code == 401

@patch("backend.app.core.security.get_supabase_client")
def test_invalid_or_expired_token_rejected(mock_get_sb):
    mock_sb = MagicMock()
    mock_sb.auth.get_user.side_effect = Exception("invalid JWT: signature is invalid")
    mock_get_sb.return_value = mock_sb

    res = client.get("/api/history", headers={"Authorization": "Bearer live-jwt-expired"})
    assert res.status_code == 401
    assert "Invalid, expired, or malformed authentication token" in res.json()["detail"]

@patch("backend.app.core.security.get_supabase_client")
def test_valid_token_verified_and_anonymous_supported(mock_get_sb):
    mock_user_obj = MagicMock()
    mock_user_obj.id = USER_A_ID
    mock_user_obj.email = "anon@test.dev"
    mock_user_obj.is_anonymous = True
    mock_user_obj.role = "authenticated"

    mock_resp = MagicMock()
    mock_resp.user = mock_user_obj

    mock_sb = MagicMock()
    mock_sb.auth.get_user.return_value = mock_resp
    mock_get_sb.return_value = mock_sb

    res = client.get("/api/history", headers={"Authorization": "Bearer live-jwt-token-a"})
    assert res.status_code == 200

# 2. Crop Ownership & Provisioning Tests
def test_default_crop_creation_and_reuse():
    repo = PredictionRepository()
    mock_sb = MagicMock()
    repo.supabase = mock_sb
    repo.use_supabase = True

    # First call: no crops exist -> insert new default crop
    mock_sb.table().select().eq().execute.return_value.data = []
    generated_crop_id = str(uuid.uuid4())
    mock_sb.table().insert().execute.return_value.data = [{"id": generated_crop_id}]

    crop_id_1 = repo.get_or_create_default_crop(USER_A_ID)
    assert crop_id_1 == generated_crop_id

    # Second call: crop already exists -> reuse existing crop
    mock_sb.table().select().eq().execute.return_value.data = [
        {"id": generated_crop_id, "crop_name": "Default Plot"}
    ]
    crop_id_2 = repo.get_or_create_default_crop(USER_A_ID)
    assert crop_id_2 == crop_id_1

def test_cross_user_crop_ownership_verification():
    repo = PredictionRepository()
    mock_sb = MagicMock()
    repo.supabase = mock_sb
    repo.use_supabase = True

    # User A owns Crop 1
    CROP_1 = str(uuid.uuid4())
    mock_sb.table().select().eq().eq().execute.return_value.data = [{"id": CROP_1}]
    assert repo.verify_crop_ownership(CROP_1, USER_A_ID) is True

    # User B does NOT own Crop 1
    mock_sb.table().select().eq().eq().execute.return_value.data = []
    assert repo.verify_crop_ownership(CROP_1, USER_B_ID) is False

from backend.app.core.config import settings

# 3. Prediction Submission & Anti-Spoofing Tests
@patch.object(settings, "DATABASE_BACKEND", "sqlite")
def test_client_cannot_spoof_user_id():
    """Client sending user_id in form data is ignored; backend uses verified token identity."""
    app.dependency_overrides[get_current_user] = lambda: user_a
    try:
        buf = _create_test_image()
        # Malicious client sends User B's ID in form data
        res = client.post(
            "/api/predict",
            files={"file": ("leaf.jpg", buf, "image/jpeg")},
            data={"model_name": "deeplabv3plus", "user_id": USER_B_ID}
        )
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "success"

        # Check history for User A contains this record
        h_res_a = client.get("/api/history")
        assert h_res_a.status_code == 200
        ids_a = [item["id"] for item in h_res_a.json()["items"]]
        assert data["prediction_id"] in ids_a
    finally:
        app.dependency_overrides.pop(get_current_user, None)

def test_cross_user_crop_access_forbidden():
    """User cannot attach prediction to another user's crop."""
    app.dependency_overrides[get_current_user] = lambda: user_b
    try:
        buf = _create_test_image()
        with patch.object(PredictionRepository, "verify_crop_ownership", return_value=False):
            res = client.post(
                "/api/predict",
                files={"file": ("leaf.jpg", buf, "image/jpeg")},
                data={"model_name": "deeplabv3plus", "crop_id": "crop-owned-by-user-a"}
            )
            assert res.status_code == 403
            assert "Forbidden" in res.json()["detail"]
    finally:
        app.dependency_overrides.pop(get_current_user, None)

# 4. History, Details, and Deletion Isolation
@patch.object(settings, "DATABASE_BACKEND", "sqlite")
def test_user_history_and_deletion_isolation():
    """User B cannot view or delete User A's prediction."""
    # Step 1: User A creates a prediction
    app.dependency_overrides[get_current_user] = lambda: user_a
    try:
        buf = _create_test_image()
        pred_res = client.post(
            "/api/predict",
            files={"file": ("leaf_a.jpg", buf, "image/jpeg")}
        )
        assert pred_res.status_code == 200
        pred_a_id = pred_res.json()["prediction_id"]
    finally:
        app.dependency_overrides.pop(get_current_user, None)

    # Step 2: User B tries to fetch User A's prediction detail -> 404
    app.dependency_overrides[get_current_user] = lambda: user_b
    try:
        detail_res = client.get(f"/api/history/{pred_a_id}")
        assert detail_res.status_code == 404

        # User B tries to delete User A's prediction -> 404
        del_res = client.delete(f"/api/history/{pred_a_id}")
        assert del_res.status_code == 404
    finally:
        app.dependency_overrides.pop(get_current_user, None)

    # Step 3: User A can view and delete their own prediction
    app.dependency_overrides[get_current_user] = lambda: user_a
    try:
        detail_res_a = client.get(f"/api/history/{pred_a_id}")
        assert detail_res_a.status_code == 200
        assert detail_res_a.json()["id"] == pred_a_id

        del_res_a = client.delete(f"/api/history/{pred_a_id}")
        assert del_res_a.status_code == 200
    finally:
        app.dependency_overrides.pop(get_current_user, None)
