import os
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.app.core.config import settings
from backend.app.db.database import Base, engine
from backend.app.api.routes import health, models, predictions, history, analytics

# Ensure DB schema created
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Intelligent Crop Health Monitoring with Pixel-Level Disease Segmentation & Severity Assessment",
    version="1.0.0"
)

# Enable CORS for React/Vite development & production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static directories (more specific prefixes first)
samples_dir = Path("data/samples")
samples_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static/samples", StaticFiles(directory=str(samples_dir)), name="samples")

static_dir = Path("backend/static")
static_dir.mkdir(parents=True, exist_ok=True)
(static_dir / "predictions").mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Include API routes
app.include_router(health.router, prefix="/api", tags=["Health"])
app.include_router(models.router, prefix="/api", tags=["Models"])
app.include_router(predictions.router, prefix="/api", tags=["Predictions"])
app.include_router(history.router, prefix="/api", tags=["History"])
app.include_router(analytics.router, prefix="/api", tags=["Analytics"])

@app.get("/")
def root():
    return {
        "project": settings.PROJECT_NAME,
        "status": "online",
        "docs_url": "/docs",
        "api_prefix": "/api"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=8000, reload=True)
