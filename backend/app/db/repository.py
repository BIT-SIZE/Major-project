import os
import json
import uuid
import logging
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import func, text

from backend.app.db.models import PredictionRecord
from backend.app.db.supabase_client import get_supabase_client
from backend.app.services.supabase_storage import SupabaseStorageService
from backend.app.core.config import settings

logger = logging.getLogger(__name__)

class SupabaseRecordAdapter:
    """
    Lightweight adapter wrapping Supabase PostgreSQL rows so that property
    access remains 100% backward-compatible with SQLAlchemy PredictionRecord.
    """
    def __init__(self, data: Dict[str, Any], storage_service: Optional[SupabaseStorageService] = None):
        self._data = data or {}
        self._storage = storage_service or SupabaseStorageService()
        self.id = self._data.get("id")
        self.user_id = self._data.get("user_id")
        self.crop_id = self._data.get("crop_id")

        # Timestamp normalization
        created_at_val = self._data.get("created_at") or self._data.get("timestamp")
        if isinstance(created_at_val, str):
            try:
                # Parse ISO format (handling Z or offset)
                clean_str = created_at_val.replace("Z", "+00:00")
                self.timestamp = datetime.fromisoformat(clean_str)
            except Exception:
                self.timestamp = datetime.now(timezone.utc)
        elif isinstance(created_at_val, datetime):
            self.timestamp = created_at_val
        else:
            self.timestamp = datetime.now(timezone.utc)

        # Core fields
        meta = self._data.get("metadata") or {}
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except Exception:
                meta = {}

        self.original_filename = self._data.get("original_filename") or meta.get("original_filename")
        self.predicted_disease = self._data.get("predicted_disease")
        self.model_name = self._data.get("model_name", "DeepLabV3+")
        self.severity_percentage = self._data.get("severity_percentage")
        self.severity_category = self._data.get("severity_category")

        # Classification fields (direct columns or metadata fallback)
        self.predicted_host = self._data.get("predicted_host") or meta.get("predicted_host")
        self.classification_score = self._data.get("classification_score") or meta.get("classification_score")
        self.classification_time_ms = float(self._data.get("classification_time_ms") or meta.get("classification_time_ms") or 0.0)

        # Top predictions
        raw_top = self._data.get("top_predictions") or self._data.get("top_predictions_json") or meta.get("top_predictions")
        if isinstance(raw_top, list):
            self.top_predictions = raw_top
            self.top_predictions_json = json.dumps(raw_top)
        elif isinstance(raw_top, str):
            self.top_predictions_json = raw_top
            try:
                self.top_predictions = json.loads(raw_top)
            except Exception:
                self.top_predictions = None
        else:
            self.top_predictions = None
            self.top_predictions_json = None

        # Severity & pixel metrics
        self.disease_pixels = int(self._data.get("disease_pixels") or meta.get("disease_pixels") or 0)
        self.leaf_pixels = int(self._data.get("leaf_pixels") or meta.get("leaf_pixels") or 0)
        self.leaf_area_available = bool(self._data.get("leaf_area_available", meta.get("leaf_area_available", False)))
        self.health_indicator = self._data.get("health_indicator") or meta.get("health_indicator")
        self.inference_time_ms = float(self._data.get("inference_time_ms") or meta.get("inference_time_ms") or 0.0)

        # Image paths & signed URLs
        raw_img = self._data.get("image_path") or self._data.get("original_image_path") or ""
        raw_mask = self._data.get("mask_path") or self._data.get("mask_image_path") or ""
        raw_overlay = self._data.get("overlay_path") or self._data.get("overlay_image_path") or meta.get("overlay_path") or ""

        # Resolve signed URLs if paths point to private Supabase storage
        if self._storage.is_available and raw_img and not raw_img.startswith("http") and not raw_img.startswith("/"):
            self.original_image_path = self._storage.create_signed_url(self._storage.bucket_images, raw_img) or raw_img
        else:
            self.original_image_path = raw_img

        if self._storage.is_available and raw_mask and not raw_mask.startswith("http") and not raw_mask.startswith("/"):
            self.mask_image_path = self._storage.create_signed_url(self._storage.bucket_masks, raw_mask) or raw_mask
        else:
            self.mask_image_path = raw_mask

        if self._storage.is_available and raw_overlay and not raw_overlay.startswith("http") and not raw_overlay.startswith("/"):
            self.overlay_image_path = self._storage.create_signed_url(self._storage.bucket_masks, raw_overlay) or raw_overlay
        else:
            self.overlay_image_path = raw_overlay

        # Raw storage paths for deletion
        self.raw_image_path = raw_img
        self.raw_mask_path = raw_mask
        self.raw_overlay_path = raw_overlay


class PredictionRepository:
    """
    Hybrid database repository:
    - Primary: Supabase PostgreSQL (when credentials configured).
    - Local Fallback: SQLite database (backend/data/crop_health.db).
    Guarantees zero downtime, offline development resilience, and safe schema adaptation.
    """
    def __init__(self, db: Optional[Session] = None):
        self.db = db
        self.storage = SupabaseStorageService()
        self.supabase = get_supabase_client()
        self.use_supabase = (
            self.supabase is not None and settings.DATABASE_BACKEND.lower() == "supabase"
        )

        if self.db is not None:
            self._ensure_sqlite_columns()

    def _ensure_sqlite_columns(self):
        """Ensures SQLite schema has classification columns."""
        if not self.db:
            return
        try:
            columns = [c[1] for c in self.db.execute(text("PRAGMA table_info(predictions);")).fetchall()]
            new_cols = [
                ("crop_id", "VARCHAR(36)"),
                ("user_id", "VARCHAR(36)"),
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

    def get_or_create_default_crop(self, user_id: str) -> str:
        """
        Finds or provisions a default crop record owned by the verified user_id.
        Guarantees that predictions have a valid crop_id foreign key reference.
        """
        if not user_id:
            raise ValueError("user_id is required to provision or locate a default crop.")

        if self.use_supabase:
            try:
                # 1. Query for existing crops owned by this user
                res = self.supabase.table("crops").select("id, crop_name").eq("user_id", user_id).execute()
                crops = res.data or []

                # Reuse 'Default Plot' if it already exists
                for c in crops:
                    if c.get("crop_name") == "Default Plot":
                        return str(c["id"])

                # Otherwise reuse any existing crop belonging to this user
                if crops:
                    return str(crops[0]["id"])

                # 2. No crop exists for this user; provision one
                crop_id = str(uuid.uuid4())
                created_at_iso = datetime.now(timezone.utc).isoformat()
                new_crop = {
                    "id": crop_id,
                    "user_id": user_id,
                    "crop_name": "Default Plot",
                    "variety": "General",
                    "location": "Primary",
                    "created_at": created_at_iso
                }
                insert_res = self.supabase.table("crops").insert(new_crop).execute()
                if insert_res.data and len(insert_res.data) > 0:
                    return str(insert_res.data[0]["id"])
                return crop_id
            except Exception as e:
                logger.error(f"Supabase get_or_create_default_crop failed: {e}")
                if not self.db:
                    raise e

        # SQLite fallback: return a stable deterministic or generated UUID
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"default-crop-{user_id}"))

    def verify_crop_ownership(self, crop_id: str, user_id: str) -> bool:
        """
        Verifies that the requested crop_id exists and is owned by the verified user_id.
        Prevents cross-user crop association.
        """
        if not crop_id or not user_id:
            return False

        if self.use_supabase:
            try:
                res = self.supabase.table("crops").select("id").eq("id", crop_id).eq("user_id", user_id).execute()
                return bool(res.data and len(res.data) > 0)
            except Exception as e:
                logger.error(f"Failed to verify crop ownership: {e}")
                return False

        # SQLite fallback: in local single-tenant mode, allow
        return True

    def create(self, record_data: Dict[str, Any]) -> Any:
        """
        Creates a new prediction record in Supabase or SQLite fallback.
        Also records individual model inference runs in model_runs.
        """
        if self.use_supabase:
            try:
                pred_id = record_data.get("id") or str(uuid.uuid4())
                created_at_iso = datetime.now(timezone.utc).isoformat()

                # Build Supabase predictions payload
                image_path = record_data.get("image_path") or record_data.get("original_image_path")
                mask_path = record_data.get("mask_path") or record_data.get("mask_image_path")
                overlay_path = record_data.get("overlay_path") or record_data.get("overlay_image_path")

                # Parse top predictions
                raw_top = record_data.get("top_predictions") or record_data.get("top_predictions_json")
                if isinstance(raw_top, str):
                    try:
                        top_preds = json.loads(raw_top)
                    except Exception:
                        top_preds = []
                elif isinstance(raw_top, list):
                    top_preds = raw_top
                else:
                    top_preds = []

                metadata_dict = {
                    "original_filename": record_data.get("original_filename"),
                    "overlay_path": overlay_path,
                    "predicted_host": record_data.get("predicted_host"),
                    "classification_score": record_data.get("classification_score"),
                    "classification_time_ms": record_data.get("classification_time_ms", 0.0),
                    "top_predictions": top_preds,
                    "health_indicator": record_data.get("health_indicator"),
                    "disease_pixels": record_data.get("disease_pixels", 0),
                    "leaf_pixels": record_data.get("leaf_pixels", 0),
                    "leaf_area_available": record_data.get("leaf_area_available", False),
                    "inference_time_ms": record_data.get("inference_time_ms", 0.0)
                }

                supabase_payload = {
                    "id": pred_id,
                    "crop_id": record_data.get("crop_id"),  # nullable
                    "user_id": record_data.get("user_id"),  # nullable
                    "image_path": image_path,
                    "mask_path": mask_path,
                    "predicted_disease": record_data.get("predicted_disease"),
                    "severity_percentage": record_data.get("severity_percentage"),
                    "severity_category": record_data.get("severity_category"),
                    "model_name": record_data.get("model_name", "DeepLabV3+"),
                    "created_at": created_at_iso
                }

                # Attempt insert with compatible extended columns first, fallback if schema is minimal
                try:
                    extended_payload = {
                        **supabase_payload,
                        "overlay_path": overlay_path,
                        "predicted_host": record_data.get("predicted_host"),
                        "classification_score": record_data.get("classification_score"),
                        "health_indicator": record_data.get("health_indicator"),
                        "top_predictions": top_preds,
                        "metadata": metadata_dict
                    }
                    res = self.supabase.table("predictions").insert(extended_payload).execute()
                except Exception as ext_err:
                    logger.debug(f"Extended Supabase insert failed ({ext_err}); trying baseline payload with metadata.")
                    try:
                        baseline_with_meta = {**supabase_payload, "metadata": metadata_dict}
                        res = self.supabase.table("predictions").insert(baseline_with_meta).execute()
                    except Exception:
                        # Fallback to strict baseline 10 columns
                        res = self.supabase.table("predictions").insert(supabase_payload).execute()

                inserted_row = res.data[0] if (res.data and len(res.data) > 0) else {**supabase_payload, "metadata": metadata_dict}

                # Log to public.model_runs (Phase 5 requirement)
                try:
                    clf_time = float(record_data.get("classification_time_ms") or 0.0)
                    total_time = float(record_data.get("inference_time_ms") or 0.0)
                    seg_time = max(0.0, total_time - clf_time)

                    runs_payload = [
                        {
                            "id": str(uuid.uuid4()),
                            "prediction_id": pred_id,
                            "model_name": "EfficientNet-B0",
                            "model_version": "1.1.0",
                            "inference_time_ms": clf_time,
                            "status": "success",
                            "created_at": created_at_iso
                        },
                        {
                            "id": str(uuid.uuid4()),
                            "prediction_id": pred_id,
                            "model_name": record_data.get("model_name", "DeepLabV3+"),
                            "model_version": "1.1.0",
                            "inference_time_ms": round(seg_time, 2),
                            "status": "success" if mask_path else "failed",
                            "created_at": created_at_iso
                        }
                    ]
                    self.supabase.table("model_runs").insert(runs_payload).execute()
                except Exception as mr_err:
                    logger.warning(f"Failed to record model_runs entry: {mr_err}")

                return SupabaseRecordAdapter(inserted_row, self.storage)

            except Exception as e:
                logger.error(f"Supabase create failed: {e}. Falling back to SQLite to protect data integrity.")
                if not self.db:
                    raise e

        # SQLite Fallback
        if self.db:
            clean_sqlite_data = {
                k: v for k, v in record_data.items()
                if hasattr(PredictionRecord, k)
            }
            record = PredictionRecord(**clean_sqlite_data)
            self.db.add(record)
            self.db.commit()
            self.db.refresh(record)
            return record

        raise RuntimeError("No database backend available.")

    def get_by_id(self, prediction_id: str, user_id: Optional[str] = None) -> Optional[Any]:
        """Retrieves a single prediction by UUID, optionally verifying user_id ownership."""
        if self.use_supabase:
            try:
                query = self.supabase.table("predictions").select("*").eq("id", prediction_id)
                if user_id:
                    query = query.eq("user_id", user_id)
                res = query.execute()
                if res.data and len(res.data) > 0:
                    return SupabaseRecordAdapter(res.data[0], self.storage)
                return None
            except Exception as e:
                logger.error(f"Supabase get_by_id failed: {e}")
                if not self.db:
                    return None

        if self.db:
            query = self.db.query(PredictionRecord).filter(PredictionRecord.id == prediction_id)
            if user_id and hasattr(PredictionRecord, "user_id"):
                query = query.filter(PredictionRecord.user_id == user_id)
            return query.first()
        return None

    def get_all(
        self,
        skip: int = 0,
        limit: int = 20,
        model_name: Optional[str] = None,
        severity_category: Optional[str] = None,
        disease_name: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> List[Any]:
        """Paginated retrieval of predictions with optional filters and user scoping."""
        if self.use_supabase:
            try:
                query = self.supabase.table("predictions").select("*")
                if user_id:
                    query = query.eq("user_id", user_id)
                if model_name:
                    query = query.ilike("model_name", f"%{model_name}%")
                if severity_category:
                    query = query.eq("severity_category", severity_category)
                if disease_name:
                    query = query.ilike("predicted_disease", f"%{disease_name}%")

                res = query.order("created_at", desc=True).range(skip, skip + limit - 1).execute()
                return [SupabaseRecordAdapter(row, self.storage) for row in (res.data or [])]
            except Exception as e:
                logger.error(f"Supabase get_all failed: {e}")
                if not self.db:
                    return []

        if self.db:
            query = self.db.query(PredictionRecord)
            if user_id and hasattr(PredictionRecord, "user_id"):
                query = query.filter(PredictionRecord.user_id == user_id)
            if model_name:
                query = query.filter(PredictionRecord.model_name.ilike(f"%{model_name}%"))
            if severity_category:
                query = query.filter(PredictionRecord.severity_category == severity_category)
            if disease_name:
                query = query.filter(PredictionRecord.predicted_disease.ilike(f"%{disease_name}%"))
            return query.order_by(PredictionRecord.timestamp.desc()).offset(skip).limit(limit).all()

        return []

    def count(
        self,
        model_name: Optional[str] = None,
        severity_category: Optional[str] = None,
        disease_name: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> int:
        """Counts total matching prediction records scoped by user."""
        if self.use_supabase:
            try:
                query = self.supabase.table("predictions").select("id", count="exact")
                if user_id:
                    query = query.eq("user_id", user_id)
                if model_name:
                    query = query.ilike("model_name", f"%{model_name}%")
                if severity_category:
                    query = query.eq("severity_category", severity_category)
                if disease_name:
                    query = query.ilike("predicted_disease", f"%{disease_name}%")
                res = query.execute()
                return res.count or 0
            except Exception as e:
                logger.error(f"Supabase count failed: {e}")
                if not self.db:
                    return 0

        if self.db:
            query = self.db.query(func.count(PredictionRecord.id))
            if user_id and hasattr(PredictionRecord, "user_id"):
                query = query.filter(PredictionRecord.user_id == user_id)
            if model_name:
                query = query.filter(PredictionRecord.model_name.ilike(f"%{model_name}%"))
            if severity_category:
                query = query.filter(PredictionRecord.severity_category == severity_category)
            if disease_name:
                query = query.filter(PredictionRecord.predicted_disease.ilike(f"%{disease_name}%"))
            return query.scalar() or 0

        return 0

    def delete(self, prediction_id: str, user_id: Optional[str] = None) -> bool:
        """Deletes prediction record and associated storage objects after verifying ownership."""
        if self.use_supabase:
            try:
                # 1. Fetch record first to verify user ownership and retrieve storage paths
                record = self.get_by_id(prediction_id, user_id=user_id)
                if user_id and not record:
                    return False

                # Delete storage artifacts if present
                if getattr(record, "raw_image_path", None):
                    self.storage.delete_file(self.storage.bucket_images, record.raw_image_path)
                if getattr(record, "raw_mask_path", None):
                    self.storage.delete_file(self.storage.bucket_masks, record.raw_mask_path)
                if getattr(record, "raw_overlay_path", None):
                    self.storage.delete_file(self.storage.bucket_masks, record.raw_overlay_path)

                # 2. Delete model runs
                try:
                    self.supabase.table("model_runs").delete().eq("prediction_id", prediction_id).execute()
                except Exception:
                    pass

                # 3. Delete prediction row
                delete_query = self.supabase.table("predictions").delete().eq("id", prediction_id)
                if user_id:
                    delete_query = delete_query.eq("user_id", user_id)
                delete_query.execute()
                return True
            except Exception as e:
                logger.error(f"Supabase delete failed: {e}")
                if not self.db:
                    return False

        if self.db:
            query = self.db.query(PredictionRecord).filter(PredictionRecord.id == prediction_id)
            if user_id and hasattr(PredictionRecord, "user_id"):
                query = query.filter(PredictionRecord.user_id == user_id)
            record = query.first()
            if not record:
                return False
            self.db.delete(record)
            self.db.commit()
            return True

        return False

    def get_analytics(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Calculates dashboard summary metrics scoped to the user's records."""
        if self.use_supabase:
            try:
                query = self.supabase.table("predictions").select("*")
                if user_id:
                    query = query.eq("user_id", user_id)
                res = query.execute()
                rows = res.data or []
                total = len(rows)

                severities = [
                    r["severity_percentage"] for r in rows
                    if r.get("severity_percentage") is not None
                ]
                avg_sev = round(sum(severities) / len(severities), 2) if severities else 0.0

                categories = {}
                models = {}
                diseases = {}
                hosts = {}
                latencies = []

                for r in rows:
                    cat = r.get("severity_category")
                    if cat:
                        categories[cat] = categories.get(cat, 0) + 1

                    mod = r.get("model_name")
                    if mod:
                        models[mod] = models.get(mod, 0) + 1

                    dis = r.get("predicted_disease")
                    if dis:
                        diseases[dis] = diseases.get(dis, 0) + 1

                    meta = r.get("metadata") or {}
                    if isinstance(meta, str):
                        try:
                            meta = json.loads(meta)
                        except Exception:
                            meta = {}

                    host = r.get("predicted_host") or meta.get("predicted_host")
                    if host:
                        hosts[host] = hosts.get(host, 0) + 1

                    lat = r.get("inference_time_ms") or meta.get("inference_time_ms")
                    if lat:
                        latencies.append(float(lat))

                top_diseases = dict(sorted(diseases.items(), key=lambda x: x[1], reverse=True)[:5])
                top_hosts = dict(sorted(hosts.items(), key=lambda x: x[1], reverse=True)[:5])
                avg_lat = round(sum(latencies) / len(latencies), 2) if latencies else 0.0

                return {
                    "total_predictions": total,
                    "average_severity_percentage": avg_sev,
                    "average_inference_time_ms": avg_lat,
                    "severity_distribution": categories,
                    "model_usage": models,
                    "top_diagnosed_diseases": top_diseases,
                    "top_diagnosed_hosts": top_hosts
                }
            except Exception as e:
                logger.error(f"Supabase get_analytics failed: {e}")
                if not self.db:
                    return {
                        "total_predictions": 0,
                        "average_severity_percentage": 0.0,
                        "average_inference_time_ms": 0.0,
                        "severity_distribution": {},
                        "model_usage": {},
                        "top_diagnosed_diseases": {},
                        "top_diagnosed_hosts": {}
                    }

        # SQLite Fallback
        if self.db:
            base_query = self.db.query(PredictionRecord)
            if user_id and hasattr(PredictionRecord, "user_id"):
                base_query = base_query.filter(PredictionRecord.user_id == user_id)

            total_predictions = base_query.count()

            sev_q = self.db.query(func.avg(PredictionRecord.severity_percentage)).filter(
                PredictionRecord.leaf_area_available == True,
                PredictionRecord.severity_percentage.isnot(None)
            )
            if user_id and hasattr(PredictionRecord, "user_id"):
                sev_q = sev_q.filter(PredictionRecord.user_id == user_id)
            avg_sev = sev_q.scalar()
            avg_sev_val = round(float(avg_sev), 2) if avg_sev is not None else 0.0

            cat_q = self.db.query(
                PredictionRecord.severity_category,
                func.count(PredictionRecord.id)
            ).filter(PredictionRecord.severity_category.isnot(None))
            if user_id and hasattr(PredictionRecord, "user_id"):
                cat_q = cat_q.filter(PredictionRecord.user_id == user_id)
            category_rows = cat_q.group_by(PredictionRecord.severity_category).all()
            categories = {row[0]: row[1] for row in category_rows}

            mod_q = self.db.query(
                PredictionRecord.model_name,
                func.count(PredictionRecord.id)
            )
            if user_id and hasattr(PredictionRecord, "user_id"):
                mod_q = mod_q.filter(PredictionRecord.user_id == user_id)
            model_rows = mod_q.group_by(PredictionRecord.model_name).all()
            models = {row[0]: row[1] for row in model_rows}

            dis_q = self.db.query(
                PredictionRecord.predicted_disease,
                func.count(PredictionRecord.id)
            ).filter(PredictionRecord.predicted_disease.isnot(None))
            if user_id and hasattr(PredictionRecord, "user_id"):
                dis_q = dis_q.filter(PredictionRecord.user_id == user_id)
            disease_rows = dis_q.group_by(PredictionRecord.predicted_disease).order_by(func.count(PredictionRecord.id).desc()).limit(5).all()
            top_diseases = {row[0]: row[1] for row in disease_rows}

            host_q = self.db.query(
                PredictionRecord.predicted_host,
                func.count(PredictionRecord.id)
            ).filter(PredictionRecord.predicted_host.isnot(None))
            if user_id and hasattr(PredictionRecord, "user_id"):
                host_q = host_q.filter(PredictionRecord.user_id == user_id)
            host_rows = host_q.group_by(PredictionRecord.predicted_host).order_by(func.count(PredictionRecord.id).desc()).limit(5).all()
            top_hosts = {row[0]: row[1] for row in host_rows}

            lat_q = self.db.query(func.avg(PredictionRecord.inference_time_ms))
            if user_id and hasattr(PredictionRecord, "user_id"):
                lat_q = lat_q.filter(PredictionRecord.user_id == user_id)
            avg_latency = lat_q.scalar() or 0.0

            return {
                "total_predictions": total_predictions,
                "average_severity_percentage": avg_sev_val,
                "average_inference_time_ms": round(float(avg_latency), 2),
                "severity_distribution": categories,
                "model_usage": models,
                "top_diagnosed_diseases": top_diseases,
                "top_diagnosed_hosts": top_hosts
            }

        return {
            "total_predictions": 0,
            "average_severity_percentage": 0.0,
            "average_inference_time_ms": 0.0,
            "severity_distribution": {},
            "model_usage": {},
            "top_diagnosed_diseases": {},
            "top_diagnosed_hosts": {}
        }
