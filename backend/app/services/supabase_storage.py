import os
import logging
from typing import Optional, Dict, Any
from backend.app.core.config import settings
from backend.app.db.supabase_client import get_supabase_client

logger = logging.getLogger(__name__)

class SupabaseStorageService:
    """
    Dedicated Supabase Cloud Storage service.
    Handles uploading images, masks, and overlays to designated buckets:
      - 'crop-images' for original field photographs.
      - 'segmentation-masks' for binary lesion masks and highlighted overlays.
    Generates collision-resistant object paths and secure signed URLs for private bucket support.
    """
    def __init__(self):
        self.bucket_images = settings.SUPABASE_STORAGE_BUCKET_IMAGES
        self.bucket_masks = settings.SUPABASE_STORAGE_BUCKET_MASKS
        self.expiry_seconds = settings.SUPABASE_SIGNED_URL_EXPIRY_SECONDS

    @property
    def is_available(self) -> bool:
        return get_supabase_client() is not None

    def upload_file(
        self,
        bucket: str,
        storage_path: str,
        file_bytes: bytes,
        content_type: str = "image/jpeg"
    ) -> Dict[str, Any]:
        """
        Uploads raw file bytes to the specified Supabase Storage bucket.
        Returns a dictionary with status, storage_path, signed_url, and error if any.
        """
        client = get_supabase_client()
        if not client:
            return {
                "success": False,
                "error": "Supabase client not initialized or credentials missing.",
                "storage_path": None,
                "signed_url": None
            }

        try:
            # Upload using Supabase Storage API
            client.storage.from_(bucket).upload(
                path=storage_path,
                file=file_bytes,
                file_options={"content-type": content_type, "upsert": "true"}
            )

            # Generate secure signed URL since buckets may be private
            signed_url = self.create_signed_url(bucket, storage_path, expires_in=self.expiry_seconds)

            return {
                "success": True,
                "bucket": bucket,
                "storage_path": storage_path,
                "signed_url": signed_url,
                "error": None
            }
        except Exception as e:
            logger.error(f"Failed to upload {storage_path} to Supabase bucket '{bucket}': {e}")
            return {
                "success": False,
                "bucket": bucket,
                "storage_path": None,
                "signed_url": None,
                "error": str(e)
            }

    def create_signed_url(
        self,
        bucket: str,
        storage_path: str,
        expires_in: Optional[int] = None
    ) -> Optional[str]:
        """
        Generates a time-limited signed URL for viewing private bucket objects.
        """
        client = get_supabase_client()
        if not client or not storage_path:
            return None

        exp = expires_in or self.expiry_seconds
        try:
            res = client.storage.from_(bucket).create_signed_url(storage_path, exp)
            if isinstance(res, dict):
                return res.get("signedURL") or res.get("signedUrl")
            elif hasattr(res, "signed_url"):
                return res.signed_url
            elif hasattr(res, "signedURL"):
                return res.signedURL
            elif hasattr(res, "signedUrl"):
                return res.signedUrl
            return str(res)
        except Exception as e:
            logger.error(f"Failed to generate signed URL for '{bucket}/{storage_path}': {e}")
            return None

    def delete_file(self, bucket: str, storage_path: str) -> bool:
        """
        Deletes an object from Supabase Storage.
        """
        client = get_supabase_client()
        if not client or not storage_path:
            return False

        try:
            client.storage.from_(bucket).remove([storage_path])
            return True
        except Exception as e:
            logger.error(f"Failed to delete '{bucket}/{storage_path}': {e}")
            return False
