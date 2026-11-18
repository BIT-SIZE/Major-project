import json
import random
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

logger = logging.getLogger(__name__)

def build_disease_class_mapping(
    metadata_csv: str = "Metadatav2.csv",
    output_json: str = "models/checkpoints/disease_classes.json"
) -> Dict[str, Any]:
    """
    Extracts all unique disease categories and their host plants from Metadatav2.csv.
    Saves a persistent, versioned JSON mapping file for deterministic training and inference.
    """
    csv_path = Path(metadata_csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_csv}")

    df = pd.read_csv(csv_path)

    # Clean and sort unique disease names
    unique_diseases = sorted(df["Disease"].dropna().unique().tolist())
    num_classes = len(unique_diseases)

    disease_to_idx = {disease: idx for idx, disease in enumerate(unique_diseases)}
    idx_to_disease = {idx: disease for idx, disease in enumerate(unique_diseases)}

    # Map disease to primary host plant (most frequent plant for that disease)
    disease_to_plant = {}
    for disease in unique_diseases:
        subset = df[df["Disease"] == disease]
        primary_plant = subset["Plant"].mode()[0]
        disease_to_plant[disease] = primary_plant

    mapping_data = {
        "num_classes": num_classes,
        "classes": unique_diseases,
        "disease_to_idx": disease_to_idx,
        "idx_to_disease": idx_to_disease,
        "disease_to_plant": disease_to_plant
    }

    out_file = Path(output_json)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(mapping_data, f, indent=2)

    print(f"[DATASET] Built mapping with {num_classes} disease classes and saved to {output_json}")
    return mapping_data


def sample_class_aware(
    records: List[Dict[str, Any]],
    max_samples: Optional[int],
    seed: int = 42
) -> List[Dict[str, Any]]:
    """
    Deterministic, class-aware stratified sampling for plant disease classification.
    Guarantees:
      1. When max_samples is None, preserves all records unmodified.
      2. When max_samples >= represented classes, every represented class is guaranteed samples.
      3. Proportional, fair distribution of remaining capacity across classes.
      4. Never samples more images than a class contains.
      5. Handles minority classes safely without discarding them.
      6. Warns explicitly if max_samples < represented classes.
      7. Completely independent of alphabetical ordering in metadata CSVs.
      8. Deterministic and reproducible via configurable random seed.
    """
    if max_samples is None or max_samples >= len(records):
        return records

    if max_samples <= 0:
        return []

    rng = random.Random(seed)

    # Group records by class_idx (or disease name as fallback)
    class_to_records = defaultdict(list)
    for r in records:
        key = r.get("class_idx", r.get("disease"))
        class_to_records[key].append(r)

    sorted_classes = sorted(class_to_records.keys())
    num_classes = len(sorted_classes)

    # Edge Case: requested budget smaller than number of represented classes
    if max_samples < num_classes:
        msg = (
            f"[SAMPLING WARNING] Requested max_samples={max_samples} is smaller than "
            f"represented classes count ({num_classes}). Complete class coverage is impossible."
        )
        logger.warning(msg)
        print(msg)

        # Fairly sample distinct classes without discarding minority classes
        chosen_classes = rng.sample(sorted_classes, max_samples)
        selected = []
        for c in chosen_classes:
            s_c = sorted(class_to_records[c], key=lambda x: x.get("filename", str(x)))
            selected.append(rng.choice(s_c))
        rng.shuffle(selected)
        return selected

    # Standard Case: max_samples >= num_classes
    # 1. Allocate initial guaranteed quota of 1 sample to every represented class
    quotas = {c: 1 for c in sorted_classes}
    remaining_budget = max_samples - num_classes
    remaining_capacities = {c: len(class_to_records[c]) - 1 for c in sorted_classes}

    # 2. Fairly distribute remaining budget proportionally to class capacities
    while remaining_budget > 0:
        eligible = [c for c in sorted_classes if remaining_capacities[c] > 0]
        if not eligible:
            break

        total_cap = sum(remaining_capacities[c] for c in eligible)
        if remaining_budget >= total_cap:
            for c in eligible:
                quotas[c] += remaining_capacities[c]
                remaining_budget -= remaining_capacities[c]
                remaining_capacities[c] = 0
            break

        allocated = 0
        remainders = []
        for c in eligible:
            share = (remaining_capacities[c] / total_cap) * remaining_budget
            add = min(int(share), remaining_capacities[c])
            quotas[c] += add
            remaining_capacities[c] -= add
            allocated += add
            remainders.append((share - int(share), c))

        remaining_budget -= allocated
        if remaining_budget > 0:
            # Distribute remaining single samples by largest fractional remainder,
            # using seeded rng for deterministic tie-breaking
            remainders.sort(key=lambda x: (x[0], rng.random()), reverse=True)
            for _, c in remainders:
                if remaining_budget <= 0:
                    break
                if remaining_capacities[c] > 0:
                    quotas[c] += 1
                    remaining_capacities[c] -= 1
                    remaining_budget -= 1

    # 3. Select samples deterministically per class
    selected_records = []
    for c in sorted_classes:
        s_c = sorted(class_to_records[c], key=lambda x: x.get("filename", str(x)))
        selected_for_c = rng.sample(s_c, quotas[c])
        selected_records.extend(selected_for_c)

    # 4. Shuffle combined records deterministically
    rng.shuffle(selected_records)
    return selected_records


class PlantDiseaseDataset(Dataset):
    """
    PyTorch Dataset for multi-class plant disease classification on PlantSeg.
    Maintains exact official benchmark splits (train, val, test).
    Supports deterministic, class-aware stratified sampling.
    """
    def __init__(
        self,
        root_dir: str = ".",
        split: str = "train",
        transform: Optional[Any] = None,
        max_samples: Optional[int] = None,
        seed: int = 42,
        mapping_data: Optional[Dict[str, Any]] = None
    ):
        self.root_dir = Path(root_dir)
        self.split = split.lower()
        self.transform = transform
        self.max_samples = max_samples
        self.seed = seed

        self.split_map = {
            "train": "Training",
            "val": "Validation",
            "test": "Test"
        }
        self.folder_map = {
            "train": "train",
            "val": "val",
            "test": "test"
        }

        # Load or generate class mapping
        mapping_file = self.root_dir / "models" / "checkpoints" / "disease_classes.json"
        if mapping_data is not None:
            self.mapping = mapping_data
        elif mapping_file.exists():
            with open(mapping_file, "r", encoding="utf-8") as f:
                self.mapping = json.load(f)
        else:
            self.mapping = build_disease_class_mapping(
                metadata_csv=str(self.root_dir / "Metadatav2.csv"),
                output_json=str(mapping_file)
            )

        self.disease_to_idx = self.mapping["disease_to_idx"]
        self.disease_to_plant = self.mapping["disease_to_plant"]
        self.image_dir = self.root_dir / "images" / self.folder_map[self.split]
        self.metadata_path = self.root_dir / "Metadatav2.csv"

        self.samples = self._load_samples(max_samples)

    def _load_samples(self, max_samples: Optional[int]) -> List[Dict[str, Any]]:
        df = pd.read_csv(self.metadata_path)
        target_split_name = self.split_map[self.split]
        split_df = df[df["Split"] == target_split_name].reset_index(drop=True)

        records = []
        for _, row in split_df.iterrows():
            img_file = str(row["Name"])
            disease_name = str(row["Disease"])
            img_path = self.image_dir / img_file

            if img_path.exists() and disease_name in self.disease_to_idx:
                records.append({
                    "image_path": str(img_path),
                    "filename": img_file,
                    "disease": disease_name,
                    "class_idx": self.disease_to_idx[disease_name],
                    "plant": str(row.get("Plant", self.disease_to_plant.get(disease_name, "Unknown"))),
                    "split": self.split
                })

        # Apply deterministic class-aware sampling
        return sample_class_aware(records, max_samples=max_samples, seed=self.seed)

    def get_distribution_summary(self, target_disease: str = "bell pepper blossom end rot") -> Dict[str, Any]:
        """
        Computes a diagnostic summary of the dataset class distribution.
        """
        all_classes = self.mapping.get("classes", [])
        counts_by_class = defaultdict(int)
        for s in self.samples:
            counts_by_class[s["disease"]] += 1

        represented_classes = [c for c in all_classes if counts_by_class[c] > 0]
        zero_sample_classes = [c for c in all_classes if counts_by_class[c] == 0]

        counts_list = [counts_by_class[c] for c in represented_classes] if represented_classes else [0]
        min_count = min(counts_list) if counts_list else 0
        max_count = max(counts_list) if counts_list else 0

        target_norm = target_disease.lower().strip()
        target_count = counts_by_class.get(target_norm, 0)

        return {
            "total_samples": len(self.samples),
            "total_represented_classes": len(represented_classes),
            "total_supported_classes": len(all_classes),
            "class_distribution": dict(counts_by_class),
            "zero_sample_classes": zero_sample_classes,
            "min_class_samples": min_count,
            "max_class_samples": max_count,
            "sampling_seed": self.seed,
            "sampling_config": f"split={self.split}, max_samples={self.max_samples}, seed={self.seed}",
            "target_disease": target_norm,
            "target_disease_present": target_count > 0,
            "target_disease_count": target_count
        }

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Dict[str, Any]]:
        record = self.samples[idx]
        image = Image.open(record["image_path"]).convert("RGB")

        if self.transform is not None:
            image_tensor = self.transform(image)
        else:
            # Default standard ImageNet transform
            default_t = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            image_tensor = default_t(image)

        return image_tensor, record["class_idx"], record


def get_classification_dataloaders(
    root_dir: str = ".",
    image_size: int = 224,
    batch_size: int = 16,
    num_workers: int = 0,
    max_train_samples: Optional[int] = None,
    max_val_samples: Optional[int] = None,
    max_test_samples: Optional[int] = None,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Constructs PyTorch DataLoaders for classification with ImageNet-standard augmentations
    and deterministic, class-aware stratified sampling.
    """
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_ds = PlantDiseaseDataset(
        root_dir=root_dir,
        split="train",
        transform=train_transform,
        max_samples=max_train_samples,
        seed=seed
    )
    val_ds = PlantDiseaseDataset(
        root_dir=root_dir,
        split="val",
        transform=eval_transform,
        max_samples=max_val_samples,
        seed=seed
    )
    test_ds = PlantDiseaseDataset(
        root_dir=root_dir,
        split="test",
        transform=eval_transform,
        max_samples=max_test_samples,
        seed=seed
    )

    return {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=False),
        "val": DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False),
        "test": DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False),
        "train_dataset": train_ds,
        "val_dataset": val_ds,
        "test_dataset": test_ds,
        "mapping": train_ds.mapping
    }
