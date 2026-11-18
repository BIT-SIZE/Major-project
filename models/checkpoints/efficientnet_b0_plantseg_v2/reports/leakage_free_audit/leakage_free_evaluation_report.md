# Leakage-Free Evaluation Report: EfficientNet-B0

Evaluated on **2182 clean test samples** (excluding 113 cross-split duplicate images).

## 1. Global Performance Comparison

| Metric | Original Old | Original New | Clean Old | Clean New | Clean Advantage (New - Old) | Leakage Impact (Clean New - Orig New) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Test Accuracy** | 3.66% | 71.02% | 3.85% | **71.13%** | **+67.28** | 0.11 |
| **Top-3 Accuracy** | 4.97% | 89.46% | 5.22% | **89.46%** | **+84.24** | 0.0 |
| **Macro Precision** | 0.0013 | 0.6337 | 0.0013 | **0.6412** | **+0.6399** | 0.0075 |
| **Macro Recall** | 0.0257 | 0.5969 | 0.0259 | **0.6029** | **+0.577** | 0.006 |
| **Macro F1** | 0.0024 | 0.5972 | 0.0025 | **0.6034** | **+0.6009** | 0.0062 |
| **Weighted F1** | 0.0038 | 0.6969 | 0.0042 | **0.6981** | **+0.6939** | 0.0012 |
| **Bell Pepper Blossom End Rot Recall** | 0.0 | 1.0 | 0.0 | **1.0** | **+1.0** | 0.0 |
| **Bell Pepper Blossom End Rot F1** | 0.0 | 0.9565217391304348 | 0.0 | **0.9545454545454546** | **+0.9545** | -0.002 |

## 2. Bell Pepper Performance on Clean Test Set

| Class Name | Clean Test Support | Old Correct | New Correct | Old Recall | New Recall | Old F1 | New F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Bell Pepper Bacterial Spot | 14 | 0 | 8 | 0.0 | **0.5714** | 0.0 | **0.64** |
| Bell Pepper Blossom End Rot | 21 | 0 | 21 | 0.0 | **1.0** | 0.0 | **0.9545** |
| Bell Pepper Frogeye Leaf Spot | 6 | 0 | 0 | 0.0 | **0.0** | 0.0 | **0.0** |
| Bell Pepper Powdery Mildew | 6 | 0 | 1 | 0.0 | **0.1667** | 0.0 | **0.2222** |
