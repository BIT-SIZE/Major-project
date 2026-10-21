import os
import sys
import time
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from ml.models.classifier import PlantDiseaseClassifier
from ml.dataset.classification_loader import get_classification_dataloaders

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate EfficientNet-B0 Disease Classifier on Test Split")
    parser.add_argument("--checkpoint", type=str, default="models/checkpoints/efficientnet_b0_best.pth")
    parser.add_argument("--max-test-samples", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--reports-dir", type=str, default="reports/evaluation")
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = ROOT_DIR / args.checkpoint

    if not ckpt_path.exists():
        print(f"[ERROR] Checkpoint not found at {ckpt_path}. Please train classifier first.")
        return

    print("=" * 60)
    print(f"EVALUATING EFFICIENTNET-B0 ON PLANTSEG TEST SET")
    print(f"Device: {device} | Test Samples: {args.max_test_samples}")
    print("=" * 60)

    # 1. Load checkpoint & metadata
    ckpt = torch.load(ckpt_path, map_location=device)
    mapping = ckpt["mapping"]
    num_classes = mapping["num_classes"]

    model = PlantDiseaseClassifier(num_classes=num_classes, pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ckpt_size_mb = round(os.path.getsize(ckpt_path) / (1024 * 1024), 2)

    # 2. Load test set
    data = get_classification_dataloaders(
        root_dir=str(ROOT_DIR),
        batch_size=args.batch_size,
        max_test_samples=args.max_test_samples
    )
    test_loader = data["test"]

    all_preds = []
    all_targets = []
    top3_correct = 0
    total_samples = 0
    inference_latencies = []

    with torch.no_grad():
        for images, targets, _ in test_loader:
            images = images.to(device)
            targets = targets.to(device)

            t0 = time.time()
            logits = model(images)
            lat = (time.time() - t0) * 1000.0 / images.size(0)
            inference_latencies.append(lat)

            probs = F.softmax(logits, dim=1)

            # Top-1
            pred_classes = torch.argmax(probs, dim=1)
            all_preds.extend(pred_classes.cpu().numpy().tolist())
            all_targets.extend(targets.cpu().numpy().tolist())

            # Top-3
            _, top3_indices = torch.topk(probs, k=min(3, num_classes), dim=1)
            for t, top3 in zip(targets, top3_indices):
                if t in top3:
                    top3_correct += 1
            total_samples += images.size(0)

    # 3. Calculate Metrics
    top1_acc = round(accuracy_score(all_targets, all_preds) * 100.0, 2)
    top3_acc = round((top3_correct / max(1, total_samples)) * 100.0, 2)
    prec, rec, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average="macro", zero_division=0)
    w_prec, w_rec, w_f1, _ = precision_recall_fscore_support(all_targets, all_preds, average="weighted", zero_division=0)
    avg_latency = round(float(np.mean(inference_latencies)), 2)

    eval_results = {
        "model_name": "EfficientNet-B0",
        "checkpoint": str(ckpt_path.name),
        "supported_classes": num_classes,
        "test_samples_evaluated": total_samples,
        "top1_accuracy_pct": top1_acc,
        "top3_accuracy_pct": top3_acc,
        "macro_precision": round(float(prec), 4),
        "macro_recall": round(float(rec), 4),
        "macro_f1": round(float(f1), 4),
        "weighted_f1": round(float(w_f1), 4),
        "avg_inference_latency_ms": avg_latency,
        "parameter_count": f"{param_count:,}",
        "checkpoint_size_mb": ckpt_size_mb,
        "device": str(device)
    }

    # 4. Save JSON and Markdown Reports
    reports_dir = ROOT_DIR / args.reports_dir
    reports_dir.mkdir(parents=True, exist_ok=True)

    json_file = reports_dir / "classifier_evaluation_report.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2)

    md_file = reports_dir / "classifier_evaluation_report.md"
    with open(md_file, "w", encoding="utf-8") as f:
        f.write("# Plant Disease Classifier Evaluation Report\n\n")
        f.write(f"Evaluated on held-out **PlantSeg Official Test Split** ({total_samples} samples).\n\n")
        f.write("| Evaluation Metric | Measured Result |\n|---|---|\n")
        f.write(f"| Model Architecture | EfficientNet-B0 |\n")
        f.write(f"| Supported Disease Classes | {num_classes} |\n")
        f.write(f"| **Top-1 Diagnosis Accuracy** | **{top1_acc}%** |\n")
        f.write(f"| **Top-3 Ranked Accuracy** | **{top3_acc}%** |\n")
        f.write(f"| Macro Precision | {eval_results['macro_precision']} |\n")
        f.write(f"| Macro Recall | {eval_results['macro_recall']} |\n")
        f.write(f"| Macro F1-Score | {eval_results['macro_f1']} |\n")
        f.write(f"| Weighted F1-Score | {eval_results['weighted_f1']} |\n")
        f.write(f"| Inference Latency | {avg_latency} ms / image |\n")
        f.write(f"| Trainable Parameters | {eval_results['parameter_count']} |\n")
        f.write(f"| Model Footprint | {ckpt_size_mb} MB |\n")
        f.write(f"| Evaluation Device | {device} |\n\n")

        f.write("## Methodological Notes\n")
        f.write("1. **Top-3 Diagnosis:** In field diagnostics, complex disease pathologies often exhibit overlapping symptoms (e.g. Early Blight vs Septoria spot). Top-3 ranked predictions provide agricultural practitioners with realistic differential diagnoses.\n")
        f.write("2. **Separation from Segmentation:** Classification predicts biological disease taxonomy, while segmentation localizes lesion pixels. These models operate complementarily.\n")

    print(f"\n[REPORT] Saved evaluation metrics to {md_file}")
    print(f"Top-1 Accuracy: {top1_acc}% | Top-3 Accuracy: {top3_acc}% | Latency: {avg_latency} ms")

if __name__ == "__main__":
    main()
