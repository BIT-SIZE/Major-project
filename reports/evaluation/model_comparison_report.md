# Model Comparison & Benchmark Report

Evaluation conducted on official PlantSeg Test split under identical resolution and preprocessing.

| Metric | U-Net | DeepLabV3+ | SegFormer-B0 |
| --- | --- | --- | --- |
| Status | Trained | Trained | Trained |
| Mean IoU | 0.2668 | 0.4403 | 0.202 |
| Dice Coefficient | 0.3719 | 0.5483 | 0.2964 |
| Precision | 0.4639 | 0.7217 | 0.2914 |
| Recall | 0.4736 | 0.5381 | 0.5495 |
| Severity MAE (%) | 14.55 | 8.2 | 28.46 |
| Latency (ms/img) | 76.14 | 81.1 | 18.37 |
| Parameter Count | 4,318,401 | 40,347,041 | 3,711,329 |
| Model Size (MB) | 49.52 MB | 462.22 MB | 42.67 MB |
| Hardware Device | cpu | cpu | cpu |
