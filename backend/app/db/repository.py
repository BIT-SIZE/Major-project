import json
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import func, text
from backend.app.db.models import PredictionRecord

class PredictionRepository:
    def __init__(self, db: Session):
        self.db = db
        self._ensure_columns_exist()

    def _ensure_columns_exist(self):
        """Ensures newly added classification columns exist in SQLite without resetting existing data."""
        try:
            columns = [c[1] for c in self.db.execute(text("PRAGMA table_info(predictions);")).fetchall()]
            new_cols = [
                ("predicted_disease", "VARCHAR(128)"),
                ("predicted_host", "VARCHAR(64)"),
                ("classification_score", "FLOAT"),
                ("classification_time_ms", "FLOAT DEFAULT 0.0"),
                ("top_predictions_json", "VARCHAR(2048)")
            ]
            for col_name, col_type in new_cols:
                if col_name not in columns:
                    self.db.execute(text(f"ALTER TABLE predictions ADD COLUMN {col_name} {col_type};"))
            self.db.commit()
        except Exception:
            self.db.rollback()

    def create(self, record_data: Dict[str, Any]) -> PredictionRecord:
        record = PredictionRecord(**record_data)
        self.db.add(record)
        self.db.commit()
        self.db.refresh(record)
        return record

    def get_by_id(self, prediction_id: str) -> Optional[PredictionRecord]:
        return self.db.query(PredictionRecord).filter(PredictionRecord.id == prediction_id).first()

    def get_all(
        self,
        skip: int = 0,
        limit: int = 20,
        model_name: Optional[str] = None,
        severity_category: Optional[str] = None,
        disease_name: Optional[str] = None
    ) -> List[PredictionRecord]:
        query = self.db.query(PredictionRecord)
        if model_name:
            query = query.filter(PredictionRecord.model_name.ilike(f"%{model_name}%"))
        if severity_category:
            query = query.filter(PredictionRecord.severity_category == severity_category)
        if disease_name:
            query = query.filter(PredictionRecord.predicted_disease.ilike(f"%{disease_name}%"))
        return query.order_by(PredictionRecord.timestamp.desc()).offset(skip).limit(limit).all()

    def count(
        self,
        model_name: Optional[str] = None,
        severity_category: Optional[str] = None,
        disease_name: Optional[str] = None
    ) -> int:
        query = self.db.query(func.count(PredictionRecord.id))
        if model_name:
            query = query.filter(PredictionRecord.model_name.ilike(f"%{model_name}%"))
        if severity_category:
            query = query.filter(PredictionRecord.severity_category == severity_category)
        if disease_name:
            query = query.filter(PredictionRecord.predicted_disease.ilike(f"%{disease_name}%"))
        return query.scalar() or 0

    def delete(self, prediction_id: str) -> bool:
        record = self.get_by_id(prediction_id)
        if not record:
            return False
        self.db.delete(record)
        self.db.commit()
        return True

    def get_analytics(self) -> Dict[str, Any]:
        total_predictions = self.db.query(func.count(PredictionRecord.id)).scalar() or 0

        # Valid average severity
        avg_sev = self.db.query(
            func.avg(PredictionRecord.severity_percentage)
        ).filter(
            PredictionRecord.leaf_area_available == True,
            PredictionRecord.severity_percentage.isnot(None)
        ).scalar()

        avg_sev_val = round(float(avg_sev), 2) if avg_sev is not None else 0.0

        # Distribution by severity category
        category_rows = self.db.query(
            PredictionRecord.severity_category,
            func.count(PredictionRecord.id)
        ).filter(PredictionRecord.severity_category.isnot(None)).group_by(PredictionRecord.severity_category).all()
        categories = {row[0]: row[1] for row in category_rows}

        # Model usage
        model_rows = self.db.query(
            PredictionRecord.model_name,
            func.count(PredictionRecord.id)
        ).group_by(PredictionRecord.model_name).all()
        models = {row[0]: row[1] for row in model_rows}

        # Top diagnosed diseases
        disease_rows = self.db.query(
            PredictionRecord.predicted_disease,
            func.count(PredictionRecord.id)
        ).filter(PredictionRecord.predicted_disease.isnot(None)).group_by(PredictionRecord.predicted_disease).order_by(func.count(PredictionRecord.id).desc()).limit(5).all()
        top_diseases = {row[0]: row[1] for row in disease_rows}

        # Top diagnosed host plants
        host_rows = self.db.query(
            PredictionRecord.predicted_host,
            func.count(PredictionRecord.id)
        ).filter(PredictionRecord.predicted_host.isnot(None)).group_by(PredictionRecord.predicted_host).order_by(func.count(PredictionRecord.id).desc()).limit(5).all()
        top_hosts = {row[0]: row[1] for row in host_rows}

        # Average latency
        avg_latency = self.db.query(func.avg(PredictionRecord.inference_time_ms)).scalar() or 0.0

        return {
            "total_predictions": total_predictions,
            "average_severity_percentage": avg_sev_val,
            "average_inference_time_ms": round(float(avg_latency), 2),
            "severity_distribution": categories,
            "model_usage": models,
            "top_diagnosed_diseases": top_diseases,
            "top_diagnosed_hosts": top_hosts
        }
