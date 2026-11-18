# EfficientNet-B0 Model Comparison Report (Old vs New)

Evaluation performed on the held-out **PlantSeg Official Test Split (2,295 images)** across all 114 classes.

## 1. Global Performance Comparison

| Metric | Existing Model | New Model (Class-Aware Retrained) |
| --- | --- | --- |
| Test Accuracy | 3.66% | **71.02%** |
| Top-3 Accuracy | 4.97% | **89.46%** |
| Macro Precision | 0.0013 | **0.6337** |
| Macro Recall | 0.0257 | **0.5969** |
| Macro F1 | 0.0024 | **0.5972** |
| Weighted F1 | 0.0038 | **0.6969** |
| Bell Pepper Blossom End Rot Recall | 0.0 | **1.0** |
| Bell Pepper Blossom End Rot F1 | 0.0 | **0.9565217391304348** |

## 2. Bell Pepper Class Breakdown (Test Set)

| Class Name | Test Images | Old Correct | New Correct | Old Recall | New Recall | Old F1 | New F1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Bell Pepper Bacterial Spot | 15 | 0 | 9 | 0.0 | 0.6 | 0.0 | 0.6667 |
| Bell Pepper Blossom End Rot | 22 | 0 | 22 | 0.0 | 1.0 | 0.0 | 0.9565 |
| Bell Pepper Frogeye Leaf Spot | 6 | 0 | 0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Bell Pepper Powdery Mildew | 6 | 0 | 1 | 0.0 | 0.1667 | 0.0 | 0.2222 |

## 3. Single Image Field Diagnostic Test

Target Image: `bell_pepper_blossom_end_rot_Bing_0164.jpg` (True: Bell Pepper Blossom End Rot)

### Existing Model Result:
- **Predicted Disease:** apple black rot
- **Predicted Host:** Apple
- **Confidence:** 84.46%
- **Top-3 Predictions:**
  - apple black rot (84.46%)
  - apple rust (9.64%)
  - apple scab (5.44%)

### New Model Result:
- **Predicted Disease:** bell pepper blossom end rot
- **Predicted Host:** Bell pepper
- **Confidence:** 96.61%
- **Top-3 Predictions:**
  - bell pepper blossom end rot (96.61%)
  - tomato late blight (0.6%)
  - citrus canker (0.45%)
