# Plant Disease Classifier Evaluation Report

Evaluated on held-out **PlantSeg Official Test Split** (60 samples).

| Evaluation Metric | Measured Result |
|---|---|
| Model Architecture | EfficientNet-B0 |
| Supported Disease Classes | 114 |
| **Top-1 Diagnosis Accuracy** | **73.33%** |
| **Top-3 Ranked Accuracy** | **100.0%** |
| Macro Precision | 0.6737 |
| Macro Recall | 0.5476 |
| Macro F1-Score | 0.5959 |
| Weighted F1-Score | 0.8007 |
| Inference Latency | 8.85 ms / image |
| Trainable Parameters | 4,153,582 |
| Model Footprint | 48.01 MB |
| Evaluation Device | cpu |

## Methodological Notes
1. **Top-3 Diagnosis:** In field diagnostics, complex disease pathologies often exhibit overlapping symptoms (e.g. Early Blight vs Septoria spot). Top-3 ranked predictions provide agricultural practitioners with realistic differential diagnoses.
2. **Separation from Segmentation:** Classification predicts biological disease taxonomy, while segmentation localizes lesion pixels. These models operate complementarily.
