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

router = APIRouter()
prediction_service = PredictionService()

@router.post("/predict", response_model=PredictionResponse)
async def predict_image(
    file: UploadFile = File(...),
    model_name: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    """
    Unified analysis endpoint:
    Executes plant disease classification, lesion segmentation, and quantitative severity assessment.
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

    # 3. Predict via Unified Service
    try:
        pred_dict = prediction_service.predict(
            image_bytes=image_bytes,
            original_filename=file.filename,
            requested_model=model_name
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image processing error: {str(e)}")

    # 4. Save record to Database with classification fields
    repo = PredictionRepository(db)
    clf_data = pred_dict["classification"]
    repo.create({
        "id": pred_dict["prediction_id"],
        "original_filename": file.filename,
        "original_image_path": pred_dict["original_image_url"],
        "overlay_image_path": pred_dict["overlay_image_url"],
        "mask_image_path": pred_dict["segmentation_image_url"],
        "predicted_disease": clf_data.get("predicted_disease"),
        "predicted_host": clf_data.get("host_plant"),
        "classification_score": clf_data.get("confidence_score"),
        "classification_time_ms": pred_dict["inference"]["classification_time_ms"],
        "top_predictions_json": json.dumps(clf_data.get("top_predictions", [])),
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
