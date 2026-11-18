import os
from pathlib import Path
from typing import Set, Optional
from pydantic import BaseModel
from dotenv import load_dotenv

# Search for .env in current directory, project root, and backend/
for env_path in [Path(".env"), Path("backend/.env"), Path(__file__).resolve().parents[3] / ".env"]:
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
        break
else:
    load_dotenv()  # Fallback to default search

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
    CORS_ORIGINS: list[str] = [
        orig.strip() for orig in os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000,*").split(",")
        if orig.strip()
    ]

    # Supabase Cloud Configuration
    SUPABASE_URL: Optional[str] = os.getenv("SUPABASE_URL")
    # Support both legacy service-role key and modern secret key naming
    SUPABASE_SERVICE_ROLE_KEY: Optional[str] = (
        os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_SECRET_KEY")
    )
    SUPABASE_STORAGE_BUCKET_IMAGES: str = os.getenv("SUPABASE_STORAGE_BUCKET_IMAGES", "crop-images")
    SUPABASE_STORAGE_BUCKET_MASKS: str = os.getenv("SUPABASE_STORAGE_BUCKET_MASKS", "segmentation-masks")
    SUPABASE_SIGNED_URL_EXPIRY_SECONDS: int = int(os.getenv("SUPABASE_SIGNED_URL_EXPIRY_SECONDS", "3600"))
    DATABASE_BACKEND: str = os.getenv("DATABASE_BACKEND", "supabase")

    @property
    def is_supabase_configured(self) -> bool:
        """Returns True only if valid, non-placeholder credentials are configured."""
        if not self.SUPABASE_URL or not self.SUPABASE_SERVICE_ROLE_KEY:
            return False
        if "your-project-id" in self.SUPABASE_URL or "your-supabase" in self.SUPABASE_SERVICE_ROLE_KEY:
            return False
        return True

settings = Settings()

# Ensure directories exist
Path(settings.UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
Path(settings.STATIC_DIR).mkdir(parents=True, exist_ok=True)
