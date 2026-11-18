import os
import time
import uuid
import json
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import cv2
import torch
import torch.nn.functional as F

from ml.models.unet import UNet
from ml.models.deeplabv3plus import DeepLabV3Plus
from ml.models.segformer import SegFormerB0
from backend.app.ml.severity import DiseaseSeverityEstimator
from backend.app.services.classification_service import ClassificationService
from backend.app.services.supabase_storage import SupabaseStorageService
from backend.app.core.config import settings

class PredictionService:
    """
    Unified production inference and post-processing service.
    Coordinates:
      1. Deep Learning Disease Classification (EfficientNet-B0 with Top-3 predictions and host identification).
      2. Pixel-Level Lesion Segmentation (DeepLabV3+, U-Net, or SegFormer-B0).
      3. Quantitative Disease Severity Estimation relative to visible plant foliage.
      4. Cloud Artifact Storage via Supabase Storage buckets ('crop-images', 'segmentation-masks').
    """
    def __init__(self, checkpoints_dir: str = "models/checkpoints"):
        self.checkpoints_dir = Path(checkpoints_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.severity_estimator = DiseaseSeverityEstimator()
        self.classification_service = ClassificationService()
        self.storage = SupabaseStorageService()
        self.loaded_models: Dict[str, torch.nn.Module] = {}
        self.active_model_name = settings.DEFAULT_MODEL

        # Output static dirs
        self.static_dir = Path(settings.STATIC_DIR)
        self.preds_dir = self.static_dir / "predictions"
        self.preds_dir.mkdir(parents=True, exist_ok=True)

        # ImageNet normalization for segmentation
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        # Preload active segmentation model on startup
        self._ensure_model_loaded(self.active_model_name)

    def _ensure_model_loaded(self, model_name: str) -> Optional[torch.nn.Module]:
        norm_name = model_name.lower().replace("-", "").replace("_", "")
        if norm_name in self.loaded_models:
            self.active_model_name = norm_name
            return self.loaded_models[norm_name]

        # Model instantiation
        if "deeplab" in norm_name:
            model = DeepLabV3Plus(num_classes=1, pretrained_backbone=False)
            ckpt_file = self.checkpoints_dir / "deeplabv3plus_best.pth"
        elif "segformer" in norm_name:
            model = SegFormerB0(num_classes=1)
            ckpt_file = self.checkpoints_dir / "segformer_best.pth"
        else:
            model = UNet(n_channels=3, n_classes=1, base_features=32)
            ckpt_file = self.checkpoints_dir / "unet_best.pth"

        if ckpt_file.exists():
            print(f"[MODEL SERVICE] Loading segmentation checkpoint: {ckpt_file} onto {self.device}")
            ckpt = torch.load(ckpt_file, map_location=self.device)
            model.load_state_dict(ckpt["state_dict"])
            model.to(self.device)
            model.eval()
            self.loaded_models[norm_name] = model
            self.active_model_name = norm_name
            return model
        else:
            print(f"[MODEL SERVICE] Checkpoint {ckpt_file} not found.")
            return None

    def set_active_model(self, model_name: str) -> bool:
        model = self._ensure_model_loaded(model_name)
        if model is not None:
            self.active_model_name = model_name.lower().replace("-", "").replace("_", "")
            return True
        return False

    def classify_only(self, image_bytes: bytes) -> Dict[str, Any]:
        """Runs standalone disease classification without segmentation."""
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError("Failed to decode uploaded image. Malformed or unsupported file.")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return self.classification_service.classify(img_rgb, top_k=3)

    def predict(
        self,
        image_bytes: bytes,
        original_filename: str,
        requested_model: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Executes unified disease classification, lesion segmentation, and severity estimation.
        """
        # Decode image
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError("Failed to decode uploaded image file. Malformed or unsupported image.")

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = img_rgb.shape[:2]

        pred_id = str(uuid.uuid4())
        timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S")

        # Save original copy
        orig_filename = f"{pred_id}_orig.jpg"
        orig_filepath = self.preds_dir / orig_filename
        cv2.imwrite(str(orig_filepath), img_bgr)

        # 1. Disease Classification (EfficientNet-B0)
        t_class_start = time.time()
        clf_result = self.classification_service.classify(img_rgb, top_k=3)
        class_latency_ms = round((time.time() - t_class_start) * 1000.0, 2)

        # 2. Disease Segmentation
        target_model_name = requested_model or self.active_model_name
        seg_model = self._ensure_model_loaded(target_model_name)

        # Fallback to available model if requested is untrained
        if seg_model is None:
            for fallback in ["deeplabv3plus", "unet", "segformer"]:
                seg_model = self._ensure_model_loaded(fallback)
                if seg_model is not None:
                    target_model_name = fallback
                    break

        seg_latency_ms = 0.0
        mask_filename = None
        overlay_filename = None
        sev_result = {
            "disease_pixels": 0,
            "leaf_pixels": 0,
            "severity_percentage": None,
            "severity_category": None,
            "leaf_area_available": False,
            "health_indicator": None,
            "message": "Segmentation model checkpoint is unavailable."
        }

        if seg_model is not None:
            # Preprocessing: resize to model dimension (e.g. 256x256)
            target_size = settings.IMAGE_SIZE
            resized_img = cv2.resize(img_rgb, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
            img_norm = (resized_img.astype(np.float32) / 255.0 - self.mean) / self.std
            tensor_img = torch.from_numpy(img_norm.transpose(2, 0, 1)).unsqueeze(0).float().to(self.device)

            # Inference
            t0 = time.time()
            with torch.no_grad():
                logits = seg_model(tensor_img)
                probs = torch.sigmoid(logits)
            seg_latency_ms = round((time.time() - t0) * 1000.0, 2)

            # Resize probability map back to original image resolution for accurate spatial analysis
            prob_map_small = probs.squeeze().cpu().numpy()
            prob_map = cv2.resize(prob_map_small, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            binary_mask = (prob_map >= 0.5).astype(np.uint8)

            # Calculate Severity
            sev_result = self.severity_estimator.estimate_severity(disease_mask=binary_mask, image_rgb=img_rgb)

            # Save Visual Artifacts:
            # 1. Binary Mask (White lesions on black background)
            mask_vis = (binary_mask * 255).astype(np.uint8)
            mask_filename = f"{pred_id}_mask.png"
            mask_filepath = self.preds_dir / mask_filename
            cv2.imwrite(str(mask_filepath), mask_vis)

            # 2. Rich Multi-channel Overlay (Red lesions + Yellow contour outline)
            overlay_bgr = img_bgr.copy()
            lesion_indices = binary_mask == 1
            overlay_bgr[lesion_indices] = (
                0.35 * overlay_bgr[lesion_indices] + 0.65 * np.array([0, 0, 235], dtype=np.float32)
            ).astype(np.uint8)

            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay_bgr, contours, -1, (0, 255, 255), 2)  # Yellow outline

            overlay_filename = f"{pred_id}_overlay.jpg"
            overlay_filepath = self.preds_dir / overlay_filename
            cv2.imwrite(str(overlay_filepath), overlay_bgr)

        total_latency_ms = round(class_latency_ms + seg_latency_ms, 2)

        # Cloud Storage Upload (Supabase Storage: crop-images, segmentation-masks)
        # Standardized path: <user_id>/<pred_id>/<filename> preserves strict user-isolation
        if user_id:
            orig_storage_path = f"{user_id}/{pred_id}/orig.jpg"
            mask_storage_path = f"{user_id}/{pred_id}/mask.png" if mask_filename else None
            overlay_storage_path = f"{user_id}/{pred_id}/overlay.jpg" if overlay_filename else None
        else:
            orig_storage_path = f"predictions/{pred_id}_orig.jpg"
            mask_storage_path = f"masks/{pred_id}_mask.png" if mask_filename else None
            overlay_storage_path = f"overlays/{pred_id}_overlay.jpg" if overlay_filename else None

        orig_url = f"/static/predictions/{orig_filename}"
        mask_url = f"/static/predictions/{mask_filename}" if mask_filename else None
        overlay_url = f"/static/predictions/{overlay_filename}" if overlay_filename else None

        storage_metadata = {
            "backend": "local",
            "image_path": orig_url,
            "mask_path": mask_url,
            "overlay_path": overlay_url
        }

        if self.storage.is_available:
            # 1. Upload original image to 'crop-images' bucket
            orig_upload = self.storage.upload_file(
                bucket=self.storage.bucket_images,
                storage_path=orig_storage_path,
                file_bytes=image_bytes,
                content_type="image/jpeg"
            )
            if orig_upload.get("success"):
                orig_url = orig_upload.get("signed_url") or orig_url
                storage_metadata["image_path"] = orig_storage_path
                storage_metadata["backend"] = "supabase"

            # 2. Upload binary mask to 'segmentation-masks' bucket
            if mask_filename and seg_model is not None:
                mask_success, mask_buf = cv2.imencode(".png", mask_vis)
                if mask_success:
                    mask_upload = self.storage.upload_file(
                        bucket=self.storage.bucket_masks,
                        storage_path=mask_storage_path,
                        file_bytes=mask_buf.tobytes(),
                        content_type="image/png"
                    )
                    if mask_upload.get("success"):
                        mask_url = mask_upload.get("signed_url") or mask_url
                        storage_metadata["mask_path"] = mask_storage_path

            # 3. Upload overlay to 'segmentation-masks' bucket
            if overlay_filename and seg_model is not None:
                overlay_success, overlay_buf = cv2.imencode(".jpg", overlay_bgr)
                if overlay_success:
                    overlay_upload = self.storage.upload_file(
                        bucket=self.storage.bucket_masks,
                        storage_path=overlay_storage_path,
                        file_bytes=overlay_buf.tobytes(),
                        content_type="image/jpeg"
                    )
                    if overlay_upload.get("success"):
                        overlay_url = overlay_upload.get("signed_url") or overlay_url
                        storage_metadata["overlay_path"] = overlay_storage_path

        # Assemble unified response structure
        response_dict = {
            "prediction_id": pred_id,
            "timestamp": timestamp_str,
            "status": "success",
            "original_image_url": orig_url,
            "storage_metadata": storage_metadata,

            # Classification component
            "classification": {
                "status": clf_result["status"],
                "predicted_disease": clf_result["predicted_disease"],
                "host_plant": clf_result["host_plant"],
                "confidence_score": clf_result["confidence_score"],
                "confidence_pct": clf_result["confidence_pct"],
                "top_predictions": clf_result["top_predictions"],
                "model_name": clf_result["model_name"],
                "uncertainty_warning": clf_result["uncertainty_warning"],
                "message": clf_result["message"]
            },

            # Segmentation component
            "segmentation": {
                "status": "success" if seg_model is not None else "untrained_model",
                "model_name": target_model_name,
                "disease_segmentation_available": seg_model is not None,
                "mask_url": mask_url,
                "overlay_url": overlay_url,
                "disease_pixels": sev_result["disease_pixels"],
                "leaf_pixels": sev_result["leaf_pixels"]
            },

            # Severity estimation component
            "severity": {
                "available": sev_result["leaf_area_available"],
                "percentage": sev_result["severity_percentage"],
                "category": sev_result["severity_category"],
                "health_indicator": sev_result["health_indicator"],
                "message": sev_result["message"]
            },

            # Inference timing
            "inference": {
                "classification_time_ms": class_latency_ms,
                "segmentation_time_ms": seg_latency_ms,
                "total_time_ms": total_latency_ms
            },

            # Backward-compatible fields
            "model_name": target_model_name,
            "disease_segmentation_available": seg_model is not None,
            "disease_severity_percentage": sev_result["severity_percentage"],
            "severity_category": sev_result["severity_category"],
            "leaf_area_available": sev_result["leaf_area_available"],
            "disease_pixels": sev_result["disease_pixels"],
            "leaf_pixels": sev_result["leaf_pixels"],
            "health_indicator": sev_result["health_indicator"],
            "segmentation_image_url": mask_url,
            "overlay_image_url": overlay_url,
            "inference_time_ms": total_latency_ms,
            "message": f"Diagnosis: {clf_result.get('predicted_disease', 'Unknown')}. Severity: {sev_result.get('severity_category', 'Inconclusive')} ({sev_result.get('severity_percentage')}%)"
        }

        return response_dict
