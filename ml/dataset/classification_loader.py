import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

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


class PlantDiseaseDataset(Dataset):
    """
    PyTorch Dataset for multi-class plant disease classification on PlantSeg.
    Maintains exact official benchmark splits (train, val, test).
    """
    def __init__(
        self,
        root_dir: str = ".",
        split: str = "train",
        transform: Optional[Any] = None,
        max_samples: Optional[int] = None,
        mapping_data: Optional[Dict[str, Any]] = None
    ):
        self.root_dir = Path(root_dir)
        self.split = split.lower()
        self.transform = transform

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

        if max_samples is not None and max_samples < len(records):
            records = records[:max_samples]

        return records

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
    max_test_samples: Optional[int] = None
) -> Dict[str, DataLoader]:
    """
    Constructs PyTorch DataLoaders for classification with ImageNet-standard augmentations.
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

    train_ds = PlantDiseaseDataset(root_dir=root_dir, split="train", transform=train_transform, max_samples=max_train_samples)
    val_ds = PlantDiseaseDataset(root_dir=root_dir, split="val", transform=eval_transform, max_samples=max_val_samples)
    test_ds = PlantDiseaseDataset(root_dir=root_dir, split="test", transform=eval_transform, max_samples=max_test_samples)

    return {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=False),
        "val": DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False),
        "test": DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False),
        "mapping": train_ds.mapping
    }
