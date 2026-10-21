import os
from pathlib import Path
from typing import Set
from pydantic import BaseModel

class Settings(BaseModel):
    PROJECT_NAME: str = "Food Security Through Intelligent Crop Health Monitoring"
    API_V1_STR: str = "/api"
    DEFAULT_MODEL: str = os.getenv("DEFAULT_MODEL", "deeplabv3plus")
    FALLBACK_MODEL: str = "unet"
    IMAGE_SIZE: int = int(os.getenv("IMAGE_SIZE", "256"))
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "backend/uploads")
    STATIC_DIR: str = os.getenv("STATIC_DIR", "backend/static")
    MAX_UPLOAD_SIZE_BYTES: int = 15 * 1024 * 1024  # 15MB
    ALLOWED_IMAGE_EXTENSIONS: Set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

settings = Settings()

# Ensure directories exist
Path(settings.UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
Path(settings.STATIC_DIR).mkdir(parents=True, exist_ok=True)
