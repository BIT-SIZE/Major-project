import os
import json
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from backend.app.db.database import get_db
from backend.app.db.repository import PredictionRepository
from backend.app.schemas.predictions import PredictionHistoryListResponse, PredictionHistoryItem

router = APIRouter()

def _parse_top_preds(raw_json: Optional[str]):
    if not raw_json:
        return None
    try:
        return json.loads(raw_json)
    except Exception:
        return None

@router.get("/history", response_model=PredictionHistoryListResponse)
def get_prediction_history(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    model_name: Optional[str] = None,
    severity_category: Optional[str] = None,
    disease_name: Optional[str] = None,
    db: Session = Depends(get_db)
):
    repo = PredictionRepository(db)
    skip = (page - 1) * limit
    total = repo.count(model_name=model_name, severity_category=severity_category, disease_name=disease_name)
    records = repo.get_all(skip=skip, limit=limit, model_name=model_name, severity_category=severity_category, disease_name=disease_name)

    items = [
        PredictionHistoryItem(
            id=r.id,
            timestamp=r.timestamp.strftime("%Y-%m-%d %H:%M:%S") if r.timestamp else "",
            original_filename=r.original_filename,
            original_image_url=r.original_image_path,
            overlay_image_url=r.overlay_image_path,
            mask_image_url=r.mask_image_path,
            model_name=r.model_name,
            predicted_disease=r.predicted_disease,
            predicted_host=r.predicted_host,
            classification_score=r.classification_score,
            severity_percentage=r.severity_percentage,
            severity_category=r.severity_category,
            leaf_area_available=r.leaf_area_available,
            health_indicator=r.health_indicator,
            inference_time_ms=r.inference_time_ms,
            top_predictions=_parse_top_preds(getattr(r, "top_predictions_json", None))
        )
        for r in records
    ]

    return {"total": total, "items": items}


@router.get("/history/{prediction_id}", response_model=PredictionHistoryItem)
def get_prediction_detail(prediction_id: str, db: Session = Depends(get_db)):
    repo = PredictionRepository(db)
    r = repo.get_by_id(prediction_id)
    if not r:
        raise HTTPException(status_code=404, detail="Prediction record not found.")

    return PredictionHistoryItem(
        id=r.id,
        timestamp=r.timestamp.strftime("%Y-%m-%d %H:%M:%S") if r.timestamp else "",
        original_filename=r.original_filename,
        original_image_url=r.original_image_path,
        overlay_image_url=r.overlay_image_path,
        mask_image_url=r.mask_image_path,
        model_name=r.model_name,
        predicted_disease=r.predicted_disease,
        predicted_host=r.predicted_host,
        classification_score=r.classification_score,
        severity_percentage=r.severity_percentage,
        severity_category=r.severity_category,
        leaf_area_available=r.leaf_area_available,
        health_indicator=r.health_indicator,
        inference_time_ms=r.inference_time_ms,
        top_predictions=_parse_top_preds(getattr(r, "top_predictions_json", None))
    )


@router.delete("/history/{prediction_id}")
def delete_prediction(prediction_id: str, db: Session = Depends(get_db)):
    repo = PredictionRepository(db)
    record = repo.get_by_id(prediction_id)
    if not record:
        raise HTTPException(status_code=404, detail="Prediction record not found.")

    # Safe file cleanup within static directory
    static_root = Path("backend").resolve()
    for path_str in [record.original_image_path, record.overlay_image_path, record.mask_image_path]:
        if path_str and path_str.startswith("/static/"):
            rel_path = path_str.replace("/static/", "static/")
            full_file_path = (static_root / rel_path).resolve()
            if str(full_file_path).startswith(str(static_root)) and full_file_path.exists():
                try:
                    os.remove(full_file_path)
                except OSError:
                    pass

    repo.delete(prediction_id)
    return {"status": "success", "message": f"Prediction {prediction_id} deleted successfully."}
