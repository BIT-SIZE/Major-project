import json
from pathlib import Path
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from backend.app.db.database import get_db
from backend.app.db.repository import PredictionRepository
from backend.app.schemas.predictions import AnalyticsSummaryResponse

router = APIRouter()

@router.get("/analytics/summary", response_model=AnalyticsSummaryResponse)
def get_analytics(db: Session = Depends(get_db)):
    repo = PredictionRepository(db)
    return repo.get_analytics()


@router.get("/dataset/summary")
def get_dataset_summary():
    report_file = Path("reports/dataset_report/dataset_inspection_report.json")
    if not report_file.exists():
        return {
            "status": "pending_inspection",
            "message": "Dataset inspection report has not yet been generated. Run scripts/inspect_dataset.py.",
            "data": None
        }

    with open(report_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    return {
        "status": "inspected",
        "data": data
    }


@router.get("/dataset/samples")
def get_dataset_samples():
    samples_dir = Path("data/samples")
    if not samples_dir.exists():
        return {"samples": []}

    sample_images = []
    for f in sorted(samples_dir.glob("sample_*.jpg")):
        sample_id = f.stem
        sample_images.append({
            "id": sample_id,
            "filename": f.name,
            "url": f"/static/samples/{f.name}"
        })

    return {"samples": sample_images}
