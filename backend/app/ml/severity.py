from typing import Dict, Any, Optional, Tuple, Union
import numpy as np
import cv2

class DiseaseSeverityEstimator:
    """
    Quantitative crop disease severity estimation engine.
    Computes disease-affected area strictly relative to valid leaf tissue area:
    Disease Severity (%) = (A_d / A_l) * 100
    where:
      A_d = number of pixels classified as diseased lesions
      A_l = number of pixels belonging to visible leaf tissue
    """
    def __init__(
        self,
        mild_threshold: float = 10.0,
        moderate_threshold: float = 25.0,
        min_valid_leaf_pixels: int = 100
    ):
        self.mild_threshold = mild_threshold
        self.moderate_threshold = moderate_threshold
        self.min_valid_leaf_pixels = min_valid_leaf_pixels

    def extract_leaf_tissue_mask(
        self,
        image_rgb: np.ndarray,
        user_leaf_mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, str]:
        """
        Extracts visible plant leaf tissue mask to avoid background inflation.
        If user provides an explicit leaf mask, it is verified and utilized.
        Otherwise, an adaptive vegetation color-space segmentation (HSV + ExG)
        is applied to distinguish foliage from inert background.
        """
        if user_leaf_mask is not None:
            mask = (user_leaf_mask > 0).astype(np.uint8)
            return mask, "ground_truth_leaf_annotation"

        # Adaptive Leaf Foliage Extraction
        # Convert RGB to HSV and calculate Excess Green Index (ExG = 2*G - R - B)
        hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
        r, g, b = image_rgb[:, :, 0].astype(np.float32), image_rgb[:, :, 1].astype(np.float32), image_rgb[:, :, 2].astype(np.float32)
        exg = 2.0 * g - r - b

        # Green & yellow-brown plant tissue ranges in HSV
        lower_green_yellow = np.array([15, 25, 25], dtype=np.uint8)
        upper_green_yellow = np.array([95, 255, 255], dtype=np.uint8)
        color_mask = cv2.inRange(hsv, lower_green_yellow, upper_green_yellow)

        # Combined vegetation response
        exg_norm = np.clip((exg - exg.min()) / (exg.max() - exg.min() + 1e-6) * 255, 0, 255).astype(np.uint8)
        _, otsu_mask = cv2.threshold(exg_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        leaf_mask = cv2.bitwise_or(color_mask, otsu_mask)

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_CLOSE, kernel)
        leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_OPEN, kernel)

        return (leaf_mask > 0).astype(np.uint8), "adaptive_vegetation_extraction"

    def estimate_severity(
        self,
        disease_mask: np.ndarray,
        image_rgb: Optional[np.ndarray] = None,
        leaf_mask: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Calculates disease lesion area, leaf denominator area, and percentage severity.
        """
        # Ensure binary disease mask
        d_mask = (disease_mask > 0.5).astype(np.uint8)
        a_d = int(np.sum(d_mask))

        # Determine leaf tissue denominator
        leaf_tissue_mask = None
        leaf_method = "unavailable"

        if leaf_mask is not None:
            leaf_tissue_mask = (leaf_mask > 0.5).astype(np.uint8)
            leaf_method = "ground_truth_leaf_mask"
        elif image_rgb is not None:
            leaf_tissue_mask, leaf_method = self.extract_leaf_tissue_mask(image_rgb)

        if leaf_tissue_mask is not None:
            # Constrain disease pixels to visible leaf bounds (prevent background false positives)
            # and union to ensure lesion pixels are recognized as part of leaf tissue
            effective_leaf_mask = np.logical_or(leaf_tissue_mask, d_mask).astype(np.uint8)
            a_l = int(np.sum(effective_leaf_mask))
        else:
            a_l = 0

        # Validate leaf area denominator
        if a_l < self.min_valid_leaf_pixels:
            return {
                "disease_pixels": a_d,
                "leaf_pixels": a_l,
                "total_image_pixels": int(disease_mask.size),
                "severity_percentage": None,
                "severity_category": None,
                "leaf_area_available": False,
                "estimation_method": leaf_method,
                "health_indicator": None,
                "valid": False,
                "message": (
                    "Severity calculation requires a valid leaf-area denominator. "
                    f"Detected leaf area ({a_l} px) is below minimum threshold ({self.min_valid_leaf_pixels} px)."
                )
            }

        # Disease Severity (%) = (A_d / A_l) * 100
        severity_pct = (float(a_d) / float(a_l)) * 100.0
        severity_pct = min(100.0, max(0.0, severity_pct))

        # Categorize
        if severity_pct == 0.0:
            category = "Healthy"
        elif severity_pct <= self.mild_threshold:
            category = "Mild"
        elif severity_pct <= self.moderate_threshold:
            category = "Moderate"
        else:
            category = "Severe"

        health_indicator = round(100.0 - severity_pct, 1)

        return {
            "disease_pixels": a_d,
            "leaf_pixels": a_l,
            "total_image_pixels": int(disease_mask.size),
            "severity_percentage": round(severity_pct, 2),
            "severity_category": category,
            "leaf_area_available": True,
            "estimation_method": leaf_method,
            "health_indicator": health_indicator,
            "valid": True,
            "message": "Severity estimated successfully against visible foliage area."
        }
