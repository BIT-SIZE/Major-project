import os
import sys
import time
import json
import hashlib
from pathlib import Path
from collections import defaultdict, Counter
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
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

def get_file_md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

class CleanTestDataset(Dataset):
    def __init__(self, records, transform=None):
        self.records = records
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        img = Image.open(rec["image_path"]).convert("RGB")
        if self.transform is not None:
            tensor = self.transform(img)
        else:
            default_t = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            tensor = default_t(img)
        return tensor, rec["class_idx"], rec

def load_classifier(ckpt_path: Path, device: torch.device):
    print(f"[LOAD] Loading checkpoint from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location=device)
    mapping = ckpt["mapping"]
    num_classes = mapping["num_classes"]
    model = PlantDiseaseClassifier(num_classes=num_classes, pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    return model, mapping, num_classes

def evaluate_model(model, loader, device, num_classes):
    all_preds = []
    all_targets = []
    top3_correct = 0
    total_samples = 0
    latencies = []

    with torch.no_grad():
        for images, targets, _ in loader:
            images = images.to(device)
            targets = targets.to(device)

            t0 = time.time()
            logits = model(images)
            lat = (time.time() - t0) * 1000.0 / images.size(0)
            latencies.append(lat)

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
    avg_latency = round(float(np.mean(latencies)), 2)

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

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 65)
    print("LEAKAGE-FREE EVALUATION: EFFICIENTNET-B0 (OLD vs NEW)")
    print(f"Device: {device}")
    print("=" * 65)

    audit_dir = ROOT_DIR / "models/checkpoints/efficientnet_b0_plantseg_v2/reports/leakage_free_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Class Mapping
    mapping_file = ROOT_DIR / "models/checkpoints/disease_classes.json"
    with open(mapping_file, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    disease_to_idx = mapping["disease_to_idx"]
    idx_to_disease = {int(k): v for k, v in mapping["idx_to_disease"].items()}
    disease_to_plant = mapping["disease_to_plant"]
    num_classes = mapping["num_classes"]

    # 2. Duplicate Identification (MD5 hashing across train and test)
    metadata_csv = ROOT_DIR / "Metadatav2.csv"
    meta_df = pd.read_csv(metadata_csv)
    train_df = meta_df[meta_df["Split"] == "Training"].copy()
    test_df = meta_df[meta_df["Split"] == "Test"].copy()

    print(f"[DATA] Computing MD5 hashes for {len(train_df)} training images...")
    train_hashes = defaultdict(list)
    for _, row in train_df.iterrows():
        img_name = str(row["Name"])
        img_p = ROOT_DIR / "images/train" / img_name
        if img_p.exists():
            h = get_file_md5(img_p)
            train_hashes[h].append({
                "filename": img_name,
                "disease": str(row["Disease"]),
                "plant": str(row["Plant"])
            })

    print(f"[DATA] Computing MD5 hashes for {len(test_df)} test images and checking overlaps...")
    duplicates = []
    clean_test_records = []
    dup_counts_by_disease = Counter()
    dup_counts_by_plant = Counter()

    for _, row in test_df.iterrows():
        img_name = str(row["Name"])
        disease_name = str(row["Disease"])
        plant_name = str(row.get("Plant", disease_to_plant.get(disease_name, "Unknown")))
        img_p = ROOT_DIR / "images/test" / img_name

        if not img_p.exists():
            continue

        h = get_file_md5(img_p)
        class_idx = disease_to_idx.get(disease_name, disease_to_idx.get(disease_name.lower()))

        if h in train_hashes:
            dup_info = {
                "test_image": img_name,
                "test_disease": disease_name,
                "test_plant": plant_name,
                "class_idx": class_idx,
                "md5": h,
                "matching_train_images": train_hashes[h]
            }
            duplicates.append(dup_info)
            dup_counts_by_disease[disease_name] += 1
            dup_counts_by_plant[plant_name] += 1
        else:
            clean_test_records.append({
                "image_path": str(img_p),
                "filename": img_name,
                "disease": disease_name,
                "plant": plant_name,
                "class_idx": class_idx,
                "md5": h
            })

    print(f"[AUDIT] Total original test images: {len(test_df)}")
    print(f"[AUDIT] Duplicate test images identified: {len(duplicates)}")
    print(f"[AUDIT] Clean leakage-free test images: {len(clean_test_records)}")

    # Save Duplicate Manifest
    dup_manifest_path = audit_dir / "duplicate_manifest.json"
    with open(dup_manifest_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_test_evaluated": len(test_df),
            "duplicate_count": len(duplicates),
            "clean_count": len(clean_test_records),
            "duplicates_by_disease": dict(dup_counts_by_disease.most_common()),
            "duplicates_by_plant": dict(dup_counts_by_plant.most_common()),
            "duplicate_pairs": duplicates
        }, f, indent=2)

    # Save Clean Test Manifest
    clean_manifest_path = audit_dir / "leakage_free_test_manifest.json"
    with open(clean_manifest_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_clean_samples": len(clean_test_records),
            "samples": clean_test_records
        }, f, indent=2)

    # 3. Build Clean Test DataLoader
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    clean_dataset = CleanTestDataset(clean_test_records, transform=eval_transform)
    clean_loader = DataLoader(clean_dataset, batch_size=32, shuffle=False, num_workers=0)

    # 4. Evaluate Existing (Old) Model
    old_ckpt_path = ROOT_DIR / "models/checkpoints/efficientnet_b0_best.pth"
    old_model, old_mapping, _ = load_classifier(old_ckpt_path, device)
    print("\n--- Evaluating Existing (Old) Model on Clean Test Subset ---")
    old_results = evaluate_model(old_model, clean_loader, device, num_classes)
    print(f"Old Model Clean Top-1 Accuracy: {old_results['top1_accuracy']}% | Macro F1: {old_results['macro_f1']}")

    # 5. Evaluate New Retrained Model
    new_ckpt_path = ROOT_DIR / "models/checkpoints/efficientnet_b0_plantseg_v2/efficientnet_b0_best.pth"
    new_model, new_mapping, _ = load_classifier(new_ckpt_path, device)
    print("\n--- Evaluating New Retrained Model on Clean Test Subset ---")
    new_results = evaluate_model(new_model, clean_loader, device, num_classes)
    print(f"New Model Clean Top-1 Accuracy: {new_results['top1_accuracy']}% | Macro F1: {new_results['macro_f1']}")

    # 6. Bell Pepper Specific Evaluation (Classes 14, 15, 16, 17)
    bell_pepper_classes = [
        (14, "Bell Pepper Bacterial Spot"),
        (15, "Bell Pepper Blossom End Rot"),
        (16, "Bell Pepper Frogeye Leaf Spot"),
        (17, "Bell Pepper Powdery Mildew")
    ]

    bp_analysis = {}
    for idx, name in bell_pepper_classes:
        clean_support = new_results["per_class"]["support"][idx]

        # Old model
        correct_old = sum(1 for t, p in zip(old_results["all_targets"], old_results["all_preds"]) if t == idx and p == idx)
        prec_old = round(old_results["per_class"]["precision"][idx], 4)
        rec_old = round(old_results["per_class"]["recall"][idx], 4)
        f1_old = round(old_results["per_class"]["f1"][idx], 4)

        # New model
        correct_new = sum(1 for t, p in zip(new_results["all_targets"], new_results["all_preds"]) if t == idx and p == idx)
        prec_new = round(new_results["per_class"]["precision"][idx], 4)
        rec_new = round(new_results["per_class"]["recall"][idx], 4)
        f1_new = round(new_results["per_class"]["f1"][idx], 4)

        bp_analysis[name] = {
            "class_index": idx,
            "clean_test_images": clean_support,
            "old_model": {
                "correct": correct_old,
                "incorrect": clean_support - correct_old,
                "precision": prec_old,
                "recall": rec_old,
                "f1_score": f1_old
            },
            "new_model": {
                "correct": correct_new,
                "incorrect": clean_support - correct_new,
                "precision": prec_new,
                "recall": rec_new,
                "f1_score": f1_new
            }
        }

    # 7. Confusion Matrix for Clean Test Set
    cm_clean = new_results["confusion_matrix"]
    np.save(audit_dir / "confusion_matrix_leakage_free.npy", cm_clean)

    plt.figure(figsize=(14, 12))
    sns.heatmap(cm_clean[:30, :30], cmap="Blues", annot=False)
    plt.title("Confusion Matrix (First 30 Classes Subset) - Clean Leakage-Free Test Set")
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.tight_layout()
    plt.savefig(audit_dir / "confusion_matrix_leakage_free.png", dpi=120)
    plt.close()

    # Classification Report
    class_names = [idx_to_disease.get(i, f"Class_{i}") for i in range(num_classes)]
    cls_report_txt = classification_report(
        new_results["all_targets"],
        new_results["all_preds"],
        labels=list(range(num_classes)),
        target_names=class_names,
        zero_division=0
    )
    with open(audit_dir / "classification_report_leakage_free.txt", "w", encoding="utf-8") as f:
        f.write(cls_report_txt)

    # 8. Load Original Comparison Report to Compute Deltas
    orig_report_path = ROOT_DIR / "models/checkpoints/efficientnet_b0_plantseg_v2/reports/model_comparison_report.json"
    with open(orig_report_path, "r", encoding="utf-8") as f:
        orig_comp = json.load(f)

    brot_idx = 15
    orig_metrics = orig_comp["metrics"]

    comparison_summary = {
        "dataset_summary": {
            "original_test_samples": len(test_df),
            "duplicates_removed": len(duplicates),
            "clean_test_samples": len(clean_test_records),
            "duplicate_percentage": round((len(duplicates) / len(test_df)) * 100.0, 2)
        },
        "metrics_comparison": {
            "Test Accuracy": {
                "original_old": orig_metrics["Test Accuracy"]["existing_model"],
                "original_new": orig_metrics["Test Accuracy"]["new_model"],
                "clean_old": f"{old_results['top1_accuracy']}%",
                "clean_new": f"{new_results['top1_accuracy']}%",
                "clean_delta": round(new_results['top1_accuracy'] - old_results['top1_accuracy'], 2),
                "new_model_impact": round(new_results['top1_accuracy'] - float(orig_metrics["Test Accuracy"]["new_model"].replace("%", "")), 2)
            },
            "Top-3 Accuracy": {
                "original_old": orig_metrics["Top-3 Accuracy"]["existing_model"],
                "original_new": orig_metrics["Top-3 Accuracy"]["new_model"],
                "clean_old": f"{old_results['top3_accuracy']}%",
                "clean_new": f"{new_results['top3_accuracy']}%",
                "clean_delta": round(new_results['top3_accuracy'] - old_results['top3_accuracy'], 2),
                "new_model_impact": round(new_results['top3_accuracy'] - float(orig_metrics["Top-3 Accuracy"]["new_model"].replace("%", "")), 2)
            },
            "Macro Precision": {
                "original_old": orig_metrics["Macro Precision"]["existing_model"],
                "original_new": orig_metrics["Macro Precision"]["new_model"],
                "clean_old": old_results["macro_precision"],
                "clean_new": new_results["macro_precision"],
                "clean_delta": round(new_results["macro_precision"] - old_results["macro_precision"], 4),
                "new_model_impact": round(new_results["macro_precision"] - orig_metrics["Macro Precision"]["new_model"], 4)
            },
            "Macro Recall": {
                "original_old": orig_metrics["Macro Recall"]["existing_model"],
                "original_new": orig_metrics["Macro Recall"]["new_model"],
                "clean_old": old_results["macro_recall"],
                "clean_new": new_results["macro_recall"],
                "clean_delta": round(new_results["macro_recall"] - old_results["macro_recall"], 4),
                "new_model_impact": round(new_results["macro_recall"] - orig_metrics["Macro Recall"]["new_model"], 4)
            },
            "Macro F1": {
                "original_old": orig_metrics["Macro F1"]["existing_model"],
                "original_new": orig_metrics["Macro F1"]["new_model"],
                "clean_old": old_results["macro_f1"],
                "clean_new": new_results["macro_f1"],
                "clean_delta": round(new_results["macro_f1"] - old_results["macro_f1"], 4),
                "new_model_impact": round(new_results["macro_f1"] - orig_metrics["Macro F1"]["new_model"], 4)
            },
            "Weighted F1": {
                "original_old": orig_metrics["Weighted F1"]["existing_model"],
                "original_new": orig_metrics["Weighted F1"]["new_model"],
                "clean_old": old_results["weighted_f1"],
                "clean_new": new_results["weighted_f1"],
                "clean_delta": round(new_results["weighted_f1"] - old_results["weighted_f1"], 4),
                "new_model_impact": round(new_results["weighted_f1"] - orig_metrics["Weighted F1"]["new_model"], 4)
            },
            "Bell Pepper Blossom End Rot Recall": {
                "original_old": orig_metrics["Bell Pepper Blossom End Rot Recall"]["existing_model"],
                "original_new": orig_metrics["Bell Pepper Blossom End Rot Recall"]["new_model"],
                "clean_old": old_results["per_class"]["recall"][brot_idx],
                "clean_new": new_results["per_class"]["recall"][brot_idx],
                "clean_delta": round(new_results["per_class"]["recall"][brot_idx] - old_results["per_class"]["recall"][brot_idx], 4),
                "new_model_impact": round(new_results["per_class"]["recall"][brot_idx] - orig_metrics["Bell Pepper Blossom End Rot Recall"]["new_model"], 4)
            },
            "Bell Pepper Blossom End Rot F1": {
                "original_old": orig_metrics["Bell Pepper Blossom End Rot F1"]["existing_model"],
                "original_new": orig_metrics["Bell Pepper Blossom End Rot F1"]["new_model"],
                "clean_old": old_results["per_class"]["f1"][brot_idx],
                "clean_new": new_results["per_class"]["f1"][brot_idx],
                "clean_delta": round(new_results["per_class"]["f1"][brot_idx] - old_results["per_class"]["f1"][brot_idx], 4),
                "new_model_impact": round(new_results["per_class"]["f1"][brot_idx] - orig_metrics["Bell Pepper Blossom End Rot F1"]["new_model"], 4)
            }
        },
        "bell_pepper_analysis": bp_analysis
    }

    # Save JSON summary
    report_json_path = audit_dir / "leakage_free_evaluation_report.json"
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(comparison_summary, f, indent=2)

    # Save Markdown report
    report_md_path = audit_dir / "leakage_free_evaluation_report.md"
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write("# Leakage-Free Evaluation Report: EfficientNet-B0\n\n")
        f.write(f"Evaluated on **{len(clean_test_records)} clean test samples** (excluding {len(duplicates)} cross-split duplicate images).\n\n")
        f.write("## 1. Global Performance Comparison\n\n")
        f.write("| Metric | Original Old | Original New | Clean Old | Clean New | Clean Advantage (New - Old) | Leakage Impact (Clean New - Orig New) |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for m_name, vals in comparison_summary["metrics_comparison"].items():
            f.write(f"| **{m_name}** | {vals['original_old']} | {vals['original_new']} | {vals['clean_old']} | **{vals['clean_new']}** | **+{vals['clean_delta']}** | {vals['new_model_impact']} |\n")

        f.write("\n## 2. Bell Pepper Performance on Clean Test Set\n\n")
        f.write("| Class Name | Clean Test Support | Old Correct | New Correct | Old Recall | New Recall | Old F1 | New F1 |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for bp_name, data in bp_analysis.items():
            f.write(f"| {bp_name} | {data['clean_test_images']} | {data['old_model']['correct']} | {data['new_model']['correct']} | {data['old_model']['recall']} | **{data['new_model']['recall']}** | {data['old_model']['f1_score']} | **{data['new_model']['f1_score']}** |\n")

    print(f"\n[REPORT] Saved evaluation outputs to {audit_dir}")
    print(f"Clean Test Accuracy - Old: {old_results['top1_accuracy']}%, New: {new_results['top1_accuracy']}%")

if __name__ == "__main__":
    main()

