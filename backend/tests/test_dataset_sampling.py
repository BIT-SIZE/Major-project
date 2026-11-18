import pytest
from pathlib import Path
from collections import Counter
import json

from ml.dataset.classification_loader import (
    sample_class_aware,
    PlantDiseaseDataset,
    get_classification_dataloaders
)

@pytest.fixture
def synthetic_records():
    """
    Creates a synthetic dataset with 114 classes having imbalanced counts
    (some with 5, 10, 50, 100 samples) in alphabetical order to test class-aware sampling.
    """
    records = []
    # 114 classes: class 0 is apple, class 15 is bell pepper blossom end rot, etc.
    for c in range(114):
        disease_name = f"disease_{c:03d}"
        if c == 15:
            disease_name = "bell pepper blossom end rot"
        elif c == 0:
            disease_name = "apple black rot"

        # Realistic imbalanced class sizes
        if c < 5:
            count = 100
        elif c < 20:
            count = 25
        elif c < 50:
            count = 10
        else:
            count = 5

        for i in range(count):
            records.append({
                "filename": f"{disease_name}_{i:04d}.jpg",
                "image_path": f"/fake/path/{disease_name}_{i:04d}.jpg",
                "disease": disease_name,
                "class_idx": c,
                "plant": "SyntheticPlant",
                "split": "train"
            })
    return records


def test_1_sampling_not_only_alphabetical(synthetic_records):
    """1. Verify sampling does not select only alphabetically initial classes."""
    budget = 200
    sampled = sample_class_aware(synthetic_records, max_samples=budget, seed=42)
    assert len(sampled) == budget

    # Classes in sampled set
    sampled_classes = {r["class_idx"] for r in sampled}
    # If it was linear slicing, classes would only be 0, 1 (first 200 items in alphabetical order)
    # With class-aware sampling, classes across the entire spectrum (e.g. > 100) are included
    high_index_classes = [c for c in sampled_classes if c >= 100]
    assert len(high_index_classes) > 0, "High-index classes must be present (not just alphabetical initial classes)"
    assert len(sampled_classes) == 114, "All 114 classes must be represented when budget (200) >= classes (114)"


def test_2_multiple_disease_categories_represented(synthetic_records):
    """2. Verify multiple disease categories are represented."""
    budget = 50  # Smaller than 114
    sampled = sample_class_aware(synthetic_records, max_samples=budget, seed=42)
    assert len(sampled) == 50
    sampled_classes = {r["class_idx"] for r in sampled}
    assert len(sampled_classes) == 50, "Should select 50 distinct disease categories when budget is 50"


def test_3_all_represented_classes_receive_samples_when_budget_permits(synthetic_records):
    """3. Verify all represented classes receive samples when budget permits (budget >= 114)."""
    for budget in [114, 250, 500]:
        sampled = sample_class_aware(synthetic_records, max_samples=budget, seed=42)
        assert len(sampled) == budget
        sampled_classes = {r["class_idx"] for r in sampled}
        assert len(sampled_classes) == 114, f"All 114 classes must be present for budget {budget}"


def test_4_sampling_reproducibility(synthetic_records):
    """4. Verify sampling is reproducible with the same random seed."""
    sample_a = sample_class_aware(synthetic_records, max_samples=300, seed=123)
    sample_b = sample_class_aware(synthetic_records, max_samples=300, seed=123)

    filenames_a = [r["filename"] for r in sample_a]
    filenames_b = [r["filename"] for r in sample_b]
    assert filenames_a == filenames_b, "Identical seed must produce identical sampled sequence"


def test_5_different_seeds_produce_different_selections(synthetic_records):
    """5. Verify different seeds produce different selections."""
    sample_a = sample_class_aware(synthetic_records, max_samples=300, seed=42)
    sample_b = sample_class_aware(synthetic_records, max_samples=300, seed=999)

    filenames_a = [r["filename"] for r in sample_a]
    filenames_b = [r["filename"] for r in sample_b]
    assert filenames_a != filenames_b, "Different seeds should produce different selections"


def test_6_no_class_exceeds_available_sample_count(synthetic_records):
    """6. Verify no class exceeds its available sample count."""
    true_counts = Counter(r["class_idx"] for r in synthetic_records)
    budget = 1000
    sampled = sample_class_aware(synthetic_records, max_samples=budget, seed=42)
    sampled_counts = Counter(r["class_idx"] for r in sampled)

    for c, count in sampled_counts.items():
        assert count <= true_counts[c], f"Class {c} sampled count ({count}) exceeds available count ({true_counts[c]})"


def test_7_max_samples_none_retains_complete_dataset(synthetic_records):
    """7. Verify max_samples=None retains the complete dataset unmodified."""
    sampled = sample_class_aware(synthetic_records, max_samples=None, seed=42)
    assert len(sampled) == len(synthetic_records)
    assert [r["filename"] for r in sampled] == [r["filename"] for r in synthetic_records]


def test_8_validation_and_test_splits_preserved():
    """8. Verify validation and test splits remain unchanged and separate."""
    # Test using actual PlantDiseaseDataset on the repository metadata
    metadata_file = Path("Metadatav2.csv")
    if metadata_file.exists():
        train_ds = PlantDiseaseDataset(root_dir=".", split="train", max_samples=None)
        val_ds = PlantDiseaseDataset(root_dir=".", split="val", max_samples=None)
        test_ds = PlantDiseaseDataset(root_dir=".", split="test", max_samples=None)

        assert len(train_ds) == 7916
        assert len(val_ds) == 1247
        assert len(test_ds) == 2295

        train_files = {s["filename"] for s in train_ds.samples}
        val_files = {s["filename"] for s in val_ds.samples}
        test_files = {s["filename"] for s in test_ds.samples}

        # Zero data leakage across splits
        assert len(train_files & val_files) == 0, "No overlap between train and val"
        assert len(train_files & test_files) == 0, "No overlap between train and test"
        assert len(val_files & test_files) == 0, "No overlap between val and test"


def test_9_class_to_index_mapping_integrity():
    """9. Verify class-to-index mapping remains exactly 114 classes and unchanged."""
    mapping_path = Path("models/checkpoints/disease_classes.json")
    assert mapping_path.exists()
    with open(mapping_path, "r", encoding="utf-8") as f:
        mapping = json.load(f)

    assert mapping["num_classes"] == 114
    assert mapping["disease_to_idx"]["bell pepper blossom end rot"] == 15
    assert mapping["classes"][15] == "bell pepper blossom end rot"
    assert mapping["disease_to_idx"]["apple black rot"] == 0
    assert mapping["classes"][0] == "apple black rot"


def test_10_bell_pepper_blossom_end_rot_selected():
    """10. Verify Bell Pepper Blossom End Rot is selected when budget >= 114."""
    # Test on real dataset metadata with a budget of 500
    dataset = PlantDiseaseDataset(root_dir=".", split="train", max_samples=500, seed=42)
    summary = dataset.get_distribution_summary("bell pepper blossom end rot")

    assert summary["target_disease_present"] is True
    assert summary["target_disease_count"] >= 1
    assert summary["total_represented_classes"] == 114
    assert len(summary["zero_sample_classes"]) == 0

