import os
import time
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from ml.models.unet import UNet
from ml.models.deeplabv3plus import DeepLabV3Plus
from ml.models.segformer import SegFormerB0
from ml.dataset.dataset_loader import get_dataloaders
from ml.evaluation.metrics import SegmentationMetricsTracker
from backend.app.ml.severity import DiseaseSeverityEstimator

class ModelEvaluatorComparator:
    """
    Standardized benchmark suite across all candidate segmentation architectures.
    Evaluates on identical test splits and measures IoU, Dice, Precision, Recall,
    Severity MAE, Inference latency, and Model footprint.
    """
    def __init__(
        self,
        root_dir: str = ".",
        checkpoint_dir: str = "models/checkpoints",
        reports_dir: str = "reports/evaluation",
        device: Optional[torch.device] = None
    ):
        self.root_dir = Path(root_dir)
        self.checkpoint_dir = self.root_dir / checkpoint_dir
        self.reports_dir = self.root_dir / reports_dir
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.severity_estimator = DiseaseSeverityEstimator()

    def _count_parameters(self, model: nn.Module) -> int:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def evaluate_model(
        self,
        architecture: str,
        checkpoint_path: Path,
        test_loader,
        image_size: int = 256
    ) -> Dict[str, Any]:
        arch = architecture.lower().replace("-", "").replace("_", "")
        if "unet" in arch:
            model = UNet(3, 1, base_features=32)
        elif "deeplab" in arch:
            model = DeepLabV3Plus(1, pretrained_backbone=False)
        elif "segformer" in arch:
            model = SegFormerB0(1)
        else:
            raise ValueError(f"Unknown model: {architecture}")

        param_count = self._count_parameters(model)
        checkpoint_size_mb = round(os.path.getsize(checkpoint_path) / (1024 * 1024), 2)

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint["state_dict"])
        model.to(self.device)
        model.eval()

        tracker = SegmentationMetricsTracker(threshold=0.5)
        inference_times = []
        severity_diffs = []

        with torch.no_grad():
            for images, targets, _ in test_loader:
                images = images.to(self.device)

                t0 = time.time()
                logits = model(images)
                t_elapsed = (time.time() - t0) * 1000.0 / images.size(0)  # ms per image
                inference_times.append(t_elapsed)

                probs = torch.sigmoid(logits)
                tracker.update(probs, targets)

                # Severity MAE against ground truth disease masks
                probs_np = probs.cpu().numpy()
                targets_np = targets.cpu().numpy()
                for p, t in zip(probs_np, targets_np):
                    p_mask = (p[0] >= 0.5).astype(np.uint8)
                    t_mask = (t[0] >= 0.5).astype(np.uint8)
                    pred_sev = (float(np.sum(p_mask)) / float(p_mask.size)) * 100.0
                    gt_sev = (float(np.sum(t_mask)) / float(t_mask.size)) * 100.0
                    severity_diffs.append(abs(pred_sev - gt_sev))

        metrics = tracker.compute()
        avg_latency = round(float(np.mean(inference_times)), 2)
        sev_mae = round(float(np.mean(severity_diffs)), 2) if severity_diffs else None

        return {
            "model_name": architecture,
            "status": "Trained",
            "mean_iou": metrics["mean_iou"],
            "dice_coefficient": metrics["mean_dice"],
            "precision": metrics["mean_precision"],
            "recall": metrics["mean_recall"],
            "severity_mae_pct": sev_mae,
            "avg_inference_latency_ms": avg_latency,
            "parameter_count": f"{param_count:,}",
            "checkpoint_size_mb": checkpoint_size_mb,
            "device": str(self.device)
        }

    def run_comparison(self, max_test_samples: int = 50, batch_size: int = 4) -> List[Dict[str, Any]]:
        print(f"\n[BENCHMARK] Running evaluation on test set ({max_test_samples} samples)...")
        loaders = get_dataloaders(
            root_dir=str(self.root_dir),
            batch_size=batch_size,
            max_test_samples=max_test_samples
        )
        test_loader = loaders["test"]

        candidate_configs = [
            ("U-Net", self.checkpoint_dir / "unet_best.pth"),
            ("DeepLabV3+", self.checkpoint_dir / "deeplabv3plus_best.pth"),
            ("SegFormer-B0", self.checkpoint_dir / "segformer_best.pth")
        ]

        results = []
        for name, ckpt_path in candidate_configs:
            if ckpt_path.exists():
                print(f"[EVAL] Evaluating {name} from {ckpt_path.name}...")
                res = self.evaluate_model(name, ckpt_path, test_loader)
                results.append(res)
            else:
                print(f"[EVAL] {name} checkpoint not found. Labeling as 'Not trained'.")
                results.append({
                    "model_name": name,
                    "status": "Not trained",
                    "mean_iou": None,
                    "dice_coefficient": None,
                    "precision": None,
                    "recall": None,
                    "severity_mae_pct": None,
                    "avg_inference_latency_ms": None,
                    "parameter_count": "N/A",
                    "checkpoint_size_mb": None,
                    "device": str(self.device)
                })

        self._save_reports(results)
        return results

    def _save_reports(self, results: List[Dict[str, Any]]):
        os.makedirs(self.reports_dir, exist_ok=True)
        # JSON
        json_path = self.reports_dir / "model_comparison_report.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({"benchmark_results": results, "generated_at": time.strftime("%Y-%m-%d %H:%M:%S")}, f, indent=2)

        # Markdown
        md_path = self.reports_dir / "model_comparison_report.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("# Model Comparison & Benchmark Report\n\n")
            f.write("Evaluation conducted on official PlantSeg Test split under identical resolution and preprocessing.\n\n")
            f.write("| Metric | " + " | ".join([r["model_name"] for r in results]) + " |\n")
            f.write("| --- | " + " | ".join(["---" for _ in results]) + " |\n")

            f.write("| Status | " + " | ".join([r["status"] for r in results]) + " |\n")
            f.write("| Mean IoU | " + " | ".join([str(r["mean_iou"]) if r["mean_iou"] is not None else "Not trained" for r in results]) + " |\n")
            f.write("| Dice Coefficient | " + " | ".join([str(r["dice_coefficient"]) if r["dice_coefficient"] is not None else "Not trained" for r in results]) + " |\n")
            f.write("| Precision | " + " | ".join([str(r["precision"]) if r["precision"] is not None else "Not trained" for r in results]) + " |\n")
            f.write("| Recall | " + " | ".join([str(r["recall"]) if r["recall"] is not None else "Not trained" for r in results]) + " |\n")
            f.write("| Severity MAE (%) | " + " | ".join([str(r["severity_mae_pct"]) if r["severity_mae_pct"] is not None else "Not trained" for r in results]) + " |\n")
            f.write("| Latency (ms/img) | " + " | ".join([str(r["avg_inference_latency_ms"]) if r["avg_inference_latency_ms"] is not None else "Not trained" for r in results]) + " |\n")
            f.write("| Parameter Count | " + " | ".join([str(r["parameter_count"]) for r in results]) + " |\n")
            f.write("| Model Size (MB) | " + " | ".join([f"{r['checkpoint_size_mb']} MB" if r["checkpoint_size_mb"] is not None else "Not trained" for r in results]) + " |\n")
            f.write("| Hardware Device | " + " | ".join([r["device"] for r in results]) + " |\n")

        # CSV
        df = pd.DataFrame(results)
        df.to_csv(self.reports_dir / "model_comparison_report.csv", index=False)
        print(f"[SAVED] Benchmark reports generated in {self.reports_dir}")
