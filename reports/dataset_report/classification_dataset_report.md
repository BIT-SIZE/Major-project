# Plant Disease Classification Dataset Report

## 1. Executive Summary
- **Total Labeled Images:** 11,458
- **Supported Disease Categories:** 114
- **Supported Host Plant Species:** 35
- **Healthy Plant Classes:** None (0 samples)
- **Important Empirical Note:** PlantSeg consists strictly of diseased plant foliage collected under natural conditions. The classifier is designed for pathology diagnosis among supported categories, not healthy/diseased binary triage.

## 2. Dataset Split Sizes
| Split | Images | Percentage | All Classes Present? |
|---|---|---|---|
| Training | 7,916 | 69.1% | Yes (114/114) |
| Test | 2,295 | 20.0% | Yes (114/114) |
| Validation | 1,247 | 10.9% | Yes (114/114) |

## 3. Class Imbalance & Representation
- **Maximum Class Count:** 390 (citrus canker)
- **Minimum Class Count:** 7 (coffee black rot)
- **Median Samples per Class:** 77
- **Mean Samples per Class:** 100.5

### Top 10 Most Frequent Disease Categories
| Disease Category | Host Plant | Sample Count |
|---|---|---|
| citrus canker | Citrus | 390 |
| wheat stripe rust | Wheat | 358 |
| wheat head scab | Wheat | 319 |
| grape downy mildew | Grape | 280 |
| wheat powdery mildew | Wheat | 271 |
| apple scab | Apple | 258 |
| soybean frog eye leaf spot | Soybean | 238 |
| bean rust | Soybean | 233 |
| wheat septoria blotch | Wheat | 232 |
| wheat loose smut | Wheat | 215 |

### Bottom 10 Least Frequent Disease Categories
| Disease Category | Host Plant | Sample Count |
|---|---|---|
| tobacco frogeye leaf spot | Tobacco | 28 |
| ginger leaf spot | Ginger | 25 |
| coffee brown eye spot | Coffee | 20 |
| carrot cercospora leaf blight | Carrot | 19 |
| raspberry leaf spot | Raspberry | 18 |
| broccoli ring spot | Broccoli | 16 |
| plum bacterial spot | Plum | 16 |
| peach anthracnose | Peach | 13 |
| peach rust | Peach | 8 |
| coffee black rot | Coffee | 7 |

## 4. Key Takeaways for Classifier Training
1. **Pretrained Weights:** Pretraining on ImageNet is essential given the fine-grained visual differences between foliar spots, rusts, and mildews.
2. **Output Head Dimension:** Exactly 114 classes dynamically aligned with the saved mapping.
3. **Top-3 Prediction Support:** Since subtle rust or blight manifestations can share visual traits, returning Top-3 ranked predictions with calibrated confidence scores gives agronomists necessary context.
