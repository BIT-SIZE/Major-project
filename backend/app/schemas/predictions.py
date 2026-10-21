from typing import Optional, Dict, Any, List
from pydantic import BaseModel

class TopPredictionItem(BaseModel):
    rank: int
    disease: str
    host_plant: str
    score: float
    confidence_pct: float

class ClassificationResult(BaseModel):
    status: str
    predicted_disease: Optional[str] = None
    host_plant: Optional[str] = None
    confidence_score: Optional[float] = None
    confidence_pct: Optional[float] = None
    top_predictions: List[TopPredictionItem] = []
    model_name: str = "EfficientNet-B0"
    uncertainty_warning: bool = False
    message: str = ""

class SegmentationResult(BaseModel):
    status: str
    model_name: str
    disease_segmentation_available: bool
    mask_url: Optional[str] = None
    overlay_url: Optional[str] = None
    disease_pixels: int = 0
    leaf_pixels: int = 0

class SeverityResult(BaseModel):
    available: bool
    percentage: Optional[float] = None
    category: Optional[str] = None
    health_indicator: Optional[float] = None
    message: str

class InferenceTiming(BaseModel):
    classification_time_ms: float = 0.0
    segmentation_time_ms: float = 0.0
    total_time_ms: float = 0.0

class PredictionResponse(BaseModel):
    prediction_id: str
    timestamp: str
    status: str
    original_image_url: str

    # Modular components
    classification: ClassificationResult
    segmentation: SegmentationResult
    severity: SeverityResult
    inference: InferenceTiming

    # Backward-compatible fields
    model_name: str
    disease_segmentation_available: bool
    disease_severity_percentage: Optional[float] = None
    severity_category: Optional[str] = None
    leaf_area_available: bool
    disease_pixels: int = 0
    leaf_pixels: int = 0
    health_indicator: Optional[float] = None
    segmentation_image_url: Optional[str] = None
    overlay_image_url: Optional[str] = None
    inference_time_ms: float = 0.0
    message: str

class ClassifyResponse(BaseModel):
    status: str
    predicted_disease: Optional[str] = None
    host_plant: Optional[str] = None
    confidence_score: Optional[float] = None
    confidence_pct: Optional[float] = None
    top_predictions: List[TopPredictionItem] = []
    model_name: str = "EfficientNet-B0"
    inference_time_ms: float = 0.0
    uncertainty_warning: bool = False
    message: str

class PredictionHistoryItem(BaseModel):
    id: str
    timestamp: str
    original_filename: Optional[str] = None
    original_image_url: str
    overlay_image_url: Optional[str] = None
    mask_image_url: Optional[str] = None
    model_name: str
    predicted_disease: Optional[str] = None
    predicted_host: Optional[str] = None
    classification_score: Optional[float] = None
    severity_percentage: Optional[float] = None
    severity_category: Optional[str] = None
    leaf_area_available: bool
    health_indicator: Optional[float] = None
    inference_time_ms: float
    top_predictions: Optional[List[Dict[str, Any]]] = None

class PredictionHistoryListResponse(BaseModel):
    total: int
    items: List[PredictionHistoryItem]

class AnalyticsSummaryResponse(BaseModel):
    total_predictions: int
    average_severity_percentage: float
    average_inference_time_ms: float
    severity_distribution: Dict[str, int]
    model_usage: Dict[str, int]
    top_diagnosed_diseases: Dict[str, int] = {}
    top_diagnosed_hosts: Dict[str, int] = {}

class ModelMetadata(BaseModel):
    name: str
    architecture: str
    task_type: str = "segmentation"  # "segmentation" or "classification"
    has_checkpoint: bool
    is_active: bool
    parameters: Optional[str] = None
    checkpoint_size_mb: Optional[float] = None
    mean_iou: Optional[float] = None
    dice_coefficient: Optional[float] = None
    accuracy_top1: Optional[float] = None
    accuracy_top3: Optional[float] = None
    avg_latency_ms: Optional[float] = None
    description: str

class ModelsListResponse(BaseModel):
    active_segmentation_model: str
    classification_model: str
    device: str
    available_models: List[ModelMetadata]
