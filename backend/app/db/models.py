import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, Float, Integer, Boolean, DateTime
from backend.app.db.database import Base

class PredictionRecord(Base):
    __tablename__ = "predictions"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)
    original_filename = Column(String(255), nullable=True)
    original_image_path = Column(String(512), nullable=False)
    overlay_image_path = Column(String(512), nullable=True)
    mask_image_path = Column(String(512), nullable=True)

    # Classification fields
    predicted_disease = Column(String(128), nullable=True, index=True)
    predicted_host = Column(String(64), nullable=True, index=True)
    classification_score = Column(Float, nullable=True)
    classification_time_ms = Column(Float, default=0.0)
    top_predictions_json = Column(String(2048), nullable=True)

    # Segmentation & Severity fields
    model_name = Column(String(64), nullable=False, index=True)
    severity_percentage = Column(Float, nullable=True)
    severity_category = Column(String(32), nullable=True, index=True)
    disease_pixels = Column(Integer, default=0)
    leaf_pixels = Column(Integer, default=0)
    leaf_area_available = Column(Boolean, default=False)
    health_indicator = Column(Float, nullable=True)
    inference_time_ms = Column(Float, default=0.0)
