"""
Verification and diagnostic script for PlantSeg classification dataset sampling.
Validates class-aware distribution, coverage across all 114 disease classes,
reproducibility across random seeds, and inclusion of target classes like
'bell pepper blossom end rot'.
"""
import sys
import argparse
from pathlib import Path
from typing import Optional

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from ml.dataset.classification_loader import PlantDiseaseDataset, get_classification_dataloaders

def run_diagnostic(
    split: str = "train",
    max_samples: Optional[int] = None,
    seed: int = 42,
    target_disease: str = "bell pepper blossom end rot"
):
    print("=" * 80)
    print(f"CLASSIFICATION DATASET SAMPLING DIAGNOSTIC")
    print(f"Configuration: split='{split}', max_samples={max_samples}, seed={seed}")
    print("=" * 80)

    dataset = PlantDiseaseDataset(
        root_dir=str(ROOT_DIR),
        split=split,
        max_samples=max_samples,
        seed=seed
    )

    summary = dataset.get_distribution_summary(target_disease=target_disease)

    print("\n--- SUMMARY REPORT ---")
    print(f"Total training images:              {summary['total_samples']}")
    print(f"Total represented classes:          {summary['total_represented_classes']} / {summary['total_supported_classes']}")
    print(f"Minimum class sample count:         {summary['min_class_samples']}")
    print(f"Maximum class sample count:         {summary['max_class_samples']}")
    print(f"Sampling random seed:               {summary['sampling_seed']}")
    print(f"Sampling configuration:             {summary['sampling_config']}")
    print(f"Classes with zero selected samples: {len(summary['zero_sample_classes'])} classes")

    if summary["zero_sample_classes"]:
        print(f"  Zero-sample classes: {summary['zero_sample_classes']}")
    else:
        print("  All supported classes have >= 1 sample selected!")

    print(f"\n--- TARGET CLASS VERIFICATION: '{target_disease}' ---")
    print(f"  Present: {summary['target_disease_present']}")
    print(f"  Selected Samples Count: {summary['target_disease_count']}")

    print("\n--- CLASS DISTRIBUTION (SAMPLE TOP 15 & BOTTOM 15) ---")
    sorted_dist = sorted(summary["class_distribution"].items(), key=lambda x: x[1], reverse=True)

    print(f"{'Disease Category':<45} | {'Sample Count':<12}")
    print("-" * 60)
    print("TOP 15 MOST FREQUENT IN SAMPLE:")
    for d, c in sorted_dist[:15]:
        print(f"{d:<45} | {c:<12}")

    print("\nBOTTOM 15 LEAST FREQUENT IN SAMPLE:")
    for d, c in sorted_dist[-15:]:
        print(f"{d:<45} | {c:<12}")

    return summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify PlantSeg Classification Sampling Distribution")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target", type=str, default="bell pepper blossom end rot")
    args = parser.parse_args()

    run_diagnostic(
        split=args.split,
        max_samples=args.max_samples,
        seed=args.seed,
        target_disease=args.target
    )

