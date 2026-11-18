import json
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import Optional

from backend.app.db.database import get_db
from backend.app.db.repository import PredictionRepository
from backend.app.services.prediction_service import PredictionService
from backend.app.schemas.predictions import PredictionResponse, ClassifyResponse
from backend.app.core.config import settings

from backend.app.core.security import get_current_user, AuthenticatedUser

router = APIRouter()
prediction_service = PredictionService()

@router.post("/predict", response_model=PredictionResponse)
async def predict_image(
    file: UploadFile = File(...),
    model_name: Optional[str] = Form(None),
    crop_id: Optional[str] = Form(None),
    current_user: AuthenticatedUser = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Unified analysis endpoint:
    Executes plant disease classification, lesion segmentation, and quantitative severity assessment.
    Stores artifacts and prediction records in Supabase (with SQLite fallback) scoped to the authenticated user.
    """
    # 1. Validation
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")

    ext = ("." + file.filename.split(".")[-1]).lower() if "." in file.filename else ""
    if ext not in settings.ALLOWED_IMAGE_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{ext}'. Supported: {list(settings.ALLOWED_IMAGE_EXTENSIONS)}"
        )

    # 2. Read bytes & check size
    image_bytes = await file.read()
    if len(image_bytes) > settings.MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(status_code=400, detail="File size exceeds maximum 15MB limit.")

    # 3. Derive verified user identity and validate/provision crop ownership
    repo = PredictionRepository(db)
    verified_user_id = current_user.id

    if crop_id:
        if not repo.verify_crop_ownership(crop_id=crop_id, user_id=verified_user_id):
            raise HTTPException(
                status_code=403,
                detail="Forbidden: The specified crop does not belong to your account."
            )
        verified_crop_id = crop_id
    else:
        verified_crop_id = repo.get_or_create_default_crop(user_id=verified_user_id)

    # 4. Predict via Unified Service with user-isolated storage paths
    try:
        pred_dict = prediction_service.predict(
            image_bytes=image_bytes,
            original_filename=file.filename,
            requested_model=model_name,
            user_id=verified_user_id
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image processing error: {str(e)}")

    # 5. Save record to Database with verified crop_id and user_id
    clf_data = pred_dict["classification"]
    storage_meta = pred_dict.get("storage_metadata", {})

    repo.create({
        "id": pred_dict["prediction_id"],
        "crop_id": verified_crop_id,
        "user_id": verified_user_id,
        "original_filename": file.filename,
        "original_image_path": pred_dict["original_image_url"],
        "overlay_image_path": pred_dict["overlay_image_url"],
        "mask_image_path": pred_dict["segmentation_image_url"],
        "image_path": storage_meta.get("image_path") or pred_dict["original_image_url"],
        "mask_path": storage_meta.get("mask_path") or pred_dict["segmentation_image_url"],
        "overlay_path": storage_meta.get("overlay_path") or pred_dict["overlay_image_url"],
        "predicted_disease": clf_data.get("predicted_disease"),
        "predicted_host": clf_data.get("host_plant"),
        "classification_score": clf_data.get("confidence_score"),
        "classification_time_ms": pred_dict["inference"]["classification_time_ms"],
        "top_predictions_json": json.dumps(clf_data.get("top_predictions", [])),
        "top_predictions": clf_data.get("top_predictions", []),
        "model_name": pred_dict["model_name"],
        "severity_percentage": pred_dict["disease_severity_percentage"],
        "severity_category": pred_dict["severity_category"],
        "disease_pixels": pred_dict["disease_pixels"],
        "leaf_pixels": pred_dict["leaf_pixels"],
        "leaf_area_available": pred_dict["leaf_area_available"],
        "health_indicator": pred_dict["health_indicator"],
        "inference_time_ms": pred_dict["inference_time_ms"]
    })

    return pred_dict


@router.post("/classify", response_model=ClassifyResponse)
async def classify_image(file: UploadFile = File(...)):
    """
    Standalone disease classification endpoint.
    Returns predicted disease category, host plant, top-3 ranked predictions, and scores.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")

    ext = ("." + file.filename.split(".")[-1]).lower() if "." in file.filename else ""
    if ext not in settings.ALLOWED_IMAGE_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported format '{ext}'.")

    image_bytes = await file.read()
    if len(image_bytes) > settings.MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(status_code=400, detail="File size exceeds 15MB.")

    try:
        clf_result = prediction_service.classify_only(image_bytes)
        return clf_result
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Classification error: {str(e)}")
