from pathlib import Path
from typing import Optional, Callable, Dict, Any, List
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

from ml.dataset.transforms import SegmentationTransforms

class PlantSegDataset(Dataset):
    """
    PyTorch Dataset implementation for PlantSeg disease segmentation.
    Supports official splits, custom sample limits, and rich metadata retrieval.
    """
    def __init__(
        self,
        root_dir: str = ".",
        split: str = "train",
        transforms: Optional[Callable] = None,
        max_samples: Optional[int] = None
    ):
        self.root_dir = Path(root_dir)
        self.split = split.lower()
        self.transforms = transforms

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

        self.image_dir = self.root_dir / "images" / self.folder_map[self.split]
        self.mask_dir = self.root_dir / "annotations" / self.folder_map[self.split]
        self.corrected_dir = self.root_dir / "corrected_annotations" / self.folder_map[self.split]
        self.metadata_path = self.root_dir / "Metadatav2.csv"

        self.samples = self._load_samples(max_samples)

    def _load_samples(self, max_samples: Optional[int]) -> List[Dict[str, Any]]:
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Missing {self.metadata_path}")

        df = pd.read_csv(self.metadata_path)
        target_split_name = self.split_map[self.split]
        split_df = df[df["Split"] == target_split_name].reset_index(drop=True)

        records = []
        for _, row in split_df.iterrows():
            img_file = str(row["Name"])
            mask_file = str(row["Label file"])

            img_path = self.image_dir / img_file
            corr_mask_path = self.corrected_dir / mask_file
            raw_mask_path = self.mask_dir / mask_file

            mask_path = corr_mask_path if corr_mask_path.exists() else raw_mask_path

            if img_path.exists() and mask_path.exists():
                records.append({
                    "image_path": str(img_path),
                    "mask_path": str(mask_path),
                    "image_name": img_file,
                    "mask_name": mask_file,
                    "plant": str(row.get("Plant", "Unknown")),
                    "disease": str(row.get("Disease", "Unknown")),
                    "split": self.split
                })

        if max_samples is not None and max_samples < len(records):
            records = records[:max_samples]

        return records

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        record = self.samples[idx]
        image = Image.open(record["image_path"]).convert("RGB")
        mask = Image.open(record["mask_path"]).convert("L")

        if self.transforms is not None:
            image_tensor, mask_tensor = self.transforms(image, mask)
        else:
            default_t = SegmentationTransforms(is_train=False)
            image_tensor, mask_tensor = default_t(image, mask)

        return image_tensor, mask_tensor, record


def get_dataloaders(
    root_dir: str = ".",
    image_size=(256, 256),
    batch_size: int = 8,
    num_workers: int = 2,
    pin_memory: Optional[bool] = None,
    persistent_workers: Optional[bool] = None,
    max_train_samples: Optional[int] = None,
    max_val_samples: Optional[int] = None,
    max_test_samples: Optional[int] = None
) -> Dict[str, DataLoader]:
    """
    Constructs PyTorch DataLoaders for train, val, and test splits.
    """
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    if persistent_workers is None:
        persistent_workers = (num_workers > 0)

    train_transforms = SegmentationTransforms(image_size=image_size, is_train=True)
    eval_transforms = SegmentationTransforms(image_size=image_size, is_train=False)

    train_ds = PlantSegDataset(root_dir=root_dir, split="train", transforms=train_transforms, max_samples=max_train_samples)
    val_ds = PlantSegDataset(root_dir=root_dir, split="val", transforms=eval_transforms, max_samples=max_val_samples)
    test_ds = PlantSegDataset(root_dir=root_dir, split="test", transforms=eval_transforms, max_samples=max_test_samples)

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = persistent_workers

    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader
    }
