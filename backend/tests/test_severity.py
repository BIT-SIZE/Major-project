import pytest
import numpy as np
from backend.app.ml.severity import DiseaseSeverityEstimator

def test_severity_calculation_known_values():
    estimator = DiseaseSeverityEstimator(mild_threshold=10.0, moderate_threshold=25.0)

    # 1000 leaf pixels, 100 disease pixels -> 10.0%
    leaf_mask = np.zeros((100, 100), dtype=np.uint8)
    leaf_mask[0:10, 0:100] = 1  # 1000 pixels

    disease_mask = np.zeros((100, 100), dtype=np.uint8)
    disease_mask[0:10, 0:10] = 1  # 100 pixels

    result = estimator.estimate_severity(disease_mask=disease_mask, leaf_mask=leaf_mask)

    assert result["valid"] is True
    assert result["disease_pixels"] == 100
    assert result["leaf_pixels"] == 1000
    assert result["severity_percentage"] == 10.0
    assert result["severity_category"] == "Mild"
    assert result["health_indicator"] == 90.0

def test_severity_healthy():
    estimator = DiseaseSeverityEstimator()
    leaf_mask = np.ones((50, 50), dtype=np.uint8)  # 2500 px
    disease_mask = np.zeros((50, 50), dtype=np.uint8)

    result = estimator.estimate_severity(disease_mask=disease_mask, leaf_mask=leaf_mask)
    assert result["valid"] is True
    assert result["disease_pixels"] == 0
    assert result["severity_percentage"] == 0.0
    assert result["severity_category"] == "Healthy"
    assert result["health_indicator"] == 100.0

def test_severity_moderate_and_severe():
    estimator = DiseaseSeverityEstimator()
    leaf_mask = np.ones((100, 100), dtype=np.uint8)  # 10000 px

    # 20% -> Moderate
    disease_mask_mod = np.zeros((100, 100), dtype=np.uint8)
    disease_mask_mod[:20, :] = 1
    res_mod = estimator.estimate_severity(disease_mask=disease_mask_mod, leaf_mask=leaf_mask)
    assert res_mod["severity_category"] == "Moderate"
    assert res_mod["severity_percentage"] == 20.0

    # 40% -> Severe
    disease_mask_sev = np.zeros((100, 100), dtype=np.uint8)
    disease_mask_sev[:40, :] = 1
    res_sev = estimator.estimate_severity(disease_mask=disease_mask_sev, leaf_mask=leaf_mask)
    assert res_sev["severity_category"] == "Severe"
    assert res_sev["severity_percentage"] == 40.0

def test_empty_or_missing_leaf_mask():
    estimator = DiseaseSeverityEstimator(min_valid_leaf_pixels=100)
    disease_mask = np.ones((50, 50), dtype=np.uint8)
    leaf_mask = np.zeros((50, 50), dtype=np.uint8)  # 0 leaf pixels

    result = estimator.estimate_severity(disease_mask=disease_mask, leaf_mask=leaf_mask)
    # Disease pixels themselves provide bounding, but if both are empty:
    empty_d = np.zeros((50, 50), dtype=np.uint8)
    res_empty = estimator.estimate_severity(disease_mask=empty_d, leaf_mask=leaf_mask)
    assert res_empty["valid"] is False
    assert res_empty["severity_percentage"] is None
    assert "below minimum threshold" in res_empty["message"]

def test_adaptive_foliage_extraction():
    estimator = DiseaseSeverityEstimator()
    # Create an RGB image with green foliage in center and dark background around
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[20:80, 20:80] = [34, 139, 34]  # Forest green

    d_mask = np.zeros((100, 100), dtype=np.uint8)
    d_mask[30:40, 30:40] = 1  # 100 diseased pixels

    result = estimator.estimate_severity(disease_mask=d_mask, image_rgb=img)
    assert result["valid"] is True
    assert result["leaf_pixels"] > 1000
    assert result["severity_percentage"] > 0.0
    assert result["estimation_method"] == "adaptive_vegetation_extraction"
