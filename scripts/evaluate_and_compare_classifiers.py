import os
import sys
import time
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn.functional as F
from torchvision import transforms
from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
    classification_report,
    confusion_matrix
)

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from ml.models.classifier import PlantDiseaseClassifier
from ml.dataset.classification_loader import PlantDiseaseDataset, get_classification_dataloaders

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate and Compare Old vs New EfficientNet-B0 on Test Split")
    parser.add_argument("--old-ckpt", type=str, default="models/checkpoints/efficientnet_b0_best.pth")
    parser.add_argument("--new-ckpt", type=str, default="models/checkpoints/efficientnet_b0_plantseg_v2/efficientnet_b0_best.pth")
    parser.add_argument("--reports-dir", type=str, default="models/checkpoints/efficientnet_b0_plantseg_v2/reports")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--single-image", type=str, default="images/train/bell_pepper_blossom_end_rot_Bing_0164.jpg")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

def load_model_from_checkpoint(ckpt_path: Path, device: torch.device):
    print(f"[LOAD] Loading checkpoint from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location=device)
    mapping = ckpt["mapping"]
    num_classes = mapping["num_classes"]

    model = PlantDiseaseClassifier(num_classes=num_classes, pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ckpt_size_mb = round(os.path.getsize(ckpt_path) / (1024 * 1024), 2)

    return model, mapping, param_count, ckpt_size_mb

def evaluate_on_test_loader(model, test_loader, device, num_classes):
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

            pred_classes = torch.argmax(probs, dim=1)
            all_preds.extend(pred_classes.cpu().numpy().tolist())
            all_targets.extend(targets.cpu().numpy().tolist())

            _, top3_indices = torch.topk(probs, k=min(3, num_classes), dim=1)
            for t, top3 in zip(targets, top3_indices):
                if t in top3:
                    top3_correct += 1
            total_samples += images.size(0)

    top1_acc = round(accuracy_score(all_targets, all_preds) * 100.0, 2)
    top3_acc = round((top3_correct / max(1, total_samples)) * 100.0, 2)

    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        all_targets, all_preds, average="macro", zero_division=0
    )
    prec_wt, rec_wt, f1_wt, _ = precision_recall_fscore_support(
        all_targets, all_preds, average="weighted", zero_division=0
    )

    prec_per_class, rec_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
        all_targets, all_preds, average=None, labels=list(range(num_classes)), zero_division=0
    )

    conf_matrix = confusion_matrix(all_targets, all_preds, labels=list(range(num_classes)))
    avg_latency = round(float(np.mean(inference_latencies)), 2)

    return {
        "all_preds": all_preds,
        "all_targets": all_targets,
        "total_samples": total_samples,
        "top1_accuracy": top1_acc,
        "top3_accuracy": top3_acc,
        "macro_precision": round(float(prec_macro), 4),
        "macro_recall": round(float(rec_macro), 4),
        "macro_f1": round(float(f1_macro), 4),
        "weighted_precision": round(float(prec_wt), 4),
        "weighted_recall": round(float(rec_wt), 4),
        "weighted_f1": round(float(f1_wt), 4),
        "avg_latency_ms": avg_latency,
        "per_class": {
            "precision": prec_per_class.tolist(),
            "recall": rec_per_class.tolist(),
            "f1": f1_per_class.tolist(),
            "support": support_per_class.tolist()
        },
        "confusion_matrix": conf_matrix
    }

def evaluate_single_image(model, image_path: Path, mapping, device):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).squeeze(0)

    top3_probs, top3_indices = torch.topk(probs, k=3)

    idx_to_disease = mapping.get("idx_to_disease", {})
    idx_map = {int(k): v for k, v in idx_to_disease.items()}
    disease_to_plant = mapping.get("disease_to_plant", {})

    predictions = []
    for prob, idx in zip(top3_probs.cpu().tolist(), top3_indices.cpu().tolist()):
        cls_name = idx_map.get(idx, f"Class {idx}")
        plant = disease_to_plant.get(cls_name, "Unknown")
        if plant == "Unknown":
            tokens = cls_name.split()
            plant = tokens[0] if tokens else "Unknown"

        predictions.append({
            "class_index": idx,
            "class_name": cls_name,
            "host": plant,
            "disease": cls_name,
            "confidence": round(prob * 100.0, 2)
        })

    top1 = predictions[0]
    return {
        "image_path": str(image_path),
        "predicted_class_index": top1["class_index"],
        "predicted_class_name": top1["class_name"],
        "predicted_host": top1["host"],
        "predicted_disease": top1["disease"],
        "confidence": top1["confidence"],
        "top3_predictions": predictions
    }

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reports_dir = ROOT_DIR / args.reports_dir
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("EVALUATION & COMPARISON: EFFICIENTNET-B0 (OLD vs NEW)")
    print(f"Device: {device}")
    print("=" * 65)

    # 1. Load test dataset (untouched test split of 2,295 images)
    data = get_classification_dataloaders(
        root_dir=str(ROOT_DIR),
        batch_size=args.batch_size,
        max_test_samples=None,
        seed=args.seed
    )
    test_loader = data["test"]
    test_dataset = data["test_dataset"]
    print(f"[DATA] Loaded test split: {len(test_dataset)} images across {data['mapping']['num_classes']} classes.")

    # 2. Load Old Model
    old_ckpt_path = ROOT_DIR / args.old_ckpt
    old_model, old_mapping, old_params, old_size = load_model_from_checkpoint(old_ckpt_path, device)

    # 3. Load New Model
    new_ckpt_path = ROOT_DIR / args.new_ckpt
    new_model, new_mapping, new_params, new_size = load_model_from_checkpoint(new_ckpt_path, device)

    num_classes = new_mapping["num_classes"]

    # 4. Evaluate Old Model on Test Set
    print("\n--- Evaluating Old Model on Test Set ---")
    old_results = evaluate_on_test_loader(old_model, test_loader, device, num_classes)
    print(f"Old Model Top-1 Accuracy: {old_results['top1_accuracy']}% | Macro F1: {old_results['macro_f1']}")

    # 5. Evaluate New Model on Test Set
    print("\n--- Evaluating New Model on Test Set ---")
    new_results = evaluate_on_test_loader(new_model, test_loader, device, num_classes)
    print(f"New Model Top-1 Accuracy: {new_results['top1_accuracy']}% | Macro F1: {new_results['macro_f1']}")

    # 6. Bell Pepper Specific Evaluation (Classes 14, 15, 16, 17)
    bell_pepper_classes = [
        (14, "Bell Pepper Bacterial Spot"),
        (15, "Bell Pepper Blossom End Rot"),
        (16, "Bell Pepper Frogeye Leaf Spot"),
        (17, "Bell Pepper Powdery Mildew")
    ]

    bp_analysis = {}
    for idx, name in bell_pepper_classes:
        # Old model
        support_old = old_results["per_class"]["support"][idx]
        prec_old = round(old_results["per_class"]["precision"][idx], 4)
        rec_old = round(old_results["per_class"]["recall"][idx], 4)
        f1_old = round(old_results["per_class"]["f1"][idx], 4)

        # Count correct and incorrect from old preds
        correct_old = sum(1 for t, p in zip(old_results["all_targets"], old_results["all_preds"]) if t == idx and p == idx)
        incorrect_old = support_old - correct_old

        # New model
        support_new = new_results["per_class"]["support"][idx]
        prec_new = round(new_results["per_class"]["precision"][idx], 4)
        rec_new = round(new_results["per_class"]["recall"][idx], 4)
        f1_new = round(new_results["per_class"]["f1"][idx], 4)

        correct_new = sum(1 for t, p in zip(new_results["all_targets"], new_results["all_preds"]) if t == idx and p == idx)
        incorrect_new = support_new - correct_new

        bp_analysis[name] = {
            "class_index": idx,
            "test_images": support_new,
            "old_model": {
                "correct": correct_old,
                "incorrect": incorrect_old,
                "precision": prec_old,
                "recall": rec_old,
                "f1_score": f1_old
            },
            "new_model": {
                "correct": correct_new,
                "incorrect": incorrect_new,
                "precision": prec_new,
                "recall": rec_new,
                "f1_score": f1_new
            }
        }

    # 7. Single-image evaluation on bell_pepper_blossom_end_rot_Bing_0164.jpg
    single_img_path = ROOT_DIR / args.single_image
    print(f"\n--- Single Image Evaluation on {single_img_path.name} ---")
    old_single = evaluate_single_image(old_model, single_img_path, old_mapping, device)
    new_single = evaluate_single_image(new_model, single_img_path, new_mapping, device)

    print(f"Old Model Prediction: {old_single['predicted_disease']} ({old_single['confidence']}%)")
    print(f"New Model Prediction: {new_single['predicted_disease']} ({new_single['confidence']}%)")

    # 8. Save Confusion Matrix Plot for New Model
    cm = new_results["confusion_matrix"]
    plt.figure(figsize=(14, 12))
    # Subsample or aggregate display for 114 classes
    sns.heatmap(cm[:30, :30], cmap="Blues", annot=False)
    plt.title("Confusion Matrix (First 30 Classes Subset) - New EfficientNet-B0")
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.tight_layout()
    cm_path = reports_dir / "confusion_matrix_new_subset.png"
    plt.savefig(cm_path, dpi=120)
    plt.close()

    # Save full confusion matrix as npy
    np.save(reports_dir / "confusion_matrix_new_full.npy", cm)

    # Save full classification report
    idx_map_new = {int(k): v for k, v in new_mapping.get("idx_to_disease", {}).items()}
    class_names = [idx_map_new.get(i, f"Class_{i}") for i in range(num_classes)]
    cls_report_txt = classification_report(
        new_results["all_targets"],
        new_results["all_preds"],
        labels=list(range(num_classes)),
        target_names=class_names,
        zero_division=0
    )
    with open(reports_dir / "classification_report_new.txt", "w", encoding="utf-8") as f:
        f.write(cls_report_txt)

    cls_report_dict = classification_report(
        new_results["all_targets"],
        new_results["all_preds"],
        labels=list(range(num_classes)),
        target_names=class_names,
        zero_division=0,
        output_dict=True
    )
    with open(reports_dir / "classification_report_new.json", "w", encoding="utf-8") as f:
        json.dump(cls_report_dict, f, indent=2)

    # 9. Comparison Table
    brot_idx = 15
    comparison_summary = {
        "metrics": {
            "Test Accuracy": {
                "existing_model": f"{old_results['top1_accuracy']}%",
                "new_model": f"{new_results['top1_accuracy']}%"
            },
            "Top-3 Accuracy": {
                "existing_model": f"{old_results['top3_accuracy']}%",
                "new_model": f"{new_results['top3_accuracy']}%"
            },
            "Macro Precision": {
                "existing_model": old_results["macro_precision"],
                "new_model": new_results["macro_precision"]
            },
            "Macro Recall": {
                "existing_model": old_results["macro_recall"],
                "new_model": new_results["macro_recall"]
            },
            "Macro F1": {
                "existing_model": old_results["macro_f1"],
                "new_model": new_results["macro_f1"]
            },
            "Weighted F1": {
                "existing_model": old_results["weighted_f1"],
                "new_model": new_results["weighted_f1"]
            },
            "Bell Pepper Blossom End Rot Recall": {
                "existing_model": old_results["per_class"]["recall"][brot_idx],
                "new_model": new_results["per_class"]["recall"][brot_idx]
            },
            "Bell Pepper Blossom End Rot F1": {
                "existing_model": old_results["per_class"]["f1"][brot_idx],
                "new_model": new_results["per_class"]["f1"][brot_idx]
            }
        },
        "bell_pepper_analysis": bp_analysis,
        "single_image_test": {
            "image": str(single_img_path),
            "old_model": old_single,
            "new_model": new_single
        }
    }

    # Save comparison report JSON
    comp_json_path = reports_dir / "model_comparison_report.json"
    with open(comp_json_path, "w", encoding="utf-8") as f:
        json.dump(comparison_summary, f, indent=2)

    # Save comparison report Markdown
    comp_md_path = reports_dir / "model_comparison_report.md"
    with open(comp_md_path, "w", encoding="utf-8") as f:
        f.write("# EfficientNet-B0 Model Comparison Report (Old vs New)\n\n")
        f.write("Evaluation performed on the held-out **PlantSeg Official Test Split (2,295 images)** across all 114 classes.\n\n")
        f.write("## 1. Global Performance Comparison\n\n")
        f.write("| Metric | Existing Model | New Model (Class-Aware Retrained) |\n")
        f.write("| --- | --- | --- |\n")
        f.write(f"| Test Accuracy | {old_results['top1_accuracy']}% | **{new_results['top1_accuracy']}%** |\n")
        f.write(f"| Top-3 Accuracy | {old_results['top3_accuracy']}% | **{new_results['top3_accuracy']}%** |\n")
        f.write(f"| Macro Precision | {old_results['macro_precision']} | **{new_results['macro_precision']}** |\n")
        f.write(f"| Macro Recall | {old_results['macro_recall']} | **{new_results['macro_recall']}** |\n")
        f.write(f"| Macro F1 | {old_results['macro_f1']} | **{new_results['macro_f1']}** |\n")
        f.write(f"| Weighted F1 | {old_results['weighted_f1']} | **{new_results['weighted_f1']}** |\n")
        f.write(f"| Bell Pepper Blossom End Rot Recall | {old_results['per_class']['recall'][brot_idx]} | **{new_results['per_class']['recall'][brot_idx]}** |\n")
        f.write(f"| Bell Pepper Blossom End Rot F1 | {old_results['per_class']['f1'][brot_idx]} | **{new_results['per_class']['f1'][brot_idx]}** |\n\n")

        f.write("## 2. Bell Pepper Class Breakdown (Test Set)\n\n")
        f.write("| Class Name | Test Images | Old Correct | New Correct | Old Recall | New Recall | Old F1 | New F1 |\n")
        f.write("| --- | --- | --- | --- | --- | --- | --- | --- |\n")
        for name, data in bp_analysis.items():
            f.write(f"| {name} | {data['test_images']} | {data['old_model']['correct']} | {data['new_model']['correct']} | {data['old_model']['recall']} | {data['new_model']['recall']} | {data['old_model']['f1_score']} | {data['new_model']['f1_score']} |\n")

        f.write("\n## 3. Single Image Field Diagnostic Test\n\n")
        f.write(f"Target Image: `{single_img_path.name}` (True: Bell Pepper Blossom End Rot)\n\n")
        f.write(f"### Existing Model Result:\n")
        f.write(f"- **Predicted Disease:** {old_single['predicted_disease']}\n")
        f.write(f"- **Predicted Host:** {old_single['predicted_host']}\n")
        f.write(f"- **Confidence:** {old_single['confidence']}%\n")
        f.write(f"- **Top-3 Predictions:**\n")
        for p in old_single['top3_predictions']:
            f.write(f"  - {p['class_name']} ({p['confidence']}%)\n")

        f.write(f"\n### New Model Result:\n")
        f.write(f"- **Predicted Disease:** {new_single['predicted_disease']}\n")
        f.write(f"- **Predicted Host:** {new_single['predicted_host']}\n")
        f.write(f"- **Confidence:** {new_single['confidence']}%\n")
        f.write(f"- **Top-3 Predictions:**\n")
        for p in new_single['top3_predictions']:
            f.write(f"  - {p['class_name']} ({p['confidence']}%)\n")

    print(f"\n[REPORT] Saved comparison report to {comp_md_path} and {comp_json_path}")

if __name__ == "__main__":
    main()

