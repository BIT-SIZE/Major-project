from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

DATASET_PATH = Path(
    r"C:\Users\hiten\Major Project\plantseg\plantsegv2"
)

MASK_DIR = DATASET_PATH / "annotations"
CORRECTED_DIR = DATASET_PATH / "corrected_annotations"

CSV_PATH = DATASET_PATH / "Metadatav2.csv"

df = pd.read_csv(CSV_PATH)

all_values = set()
mask_count = 0
value_counts = {}

print("Inspecting PlantSeg annotation masks...")

for _, row in df.iterrows():

    mask_name = str(row["Label file"])

    matches = list(MASK_DIR.rglob(mask_name))

    if len(matches) != 1:
        print("Skipping missing or duplicate mask:", mask_name)
        continue

    original_path = matches[0]

    relative_path = original_path.relative_to(MASK_DIR)

    corrected_path = CORRECTED_DIR / relative_path

    if corrected_path.exists():
        mask_path = corrected_path
    else:
        mask_path = original_path

    with Image.open(mask_path) as mask:
        mask_array = np.asarray(mask)

    values, counts = np.unique(
        mask_array,
        return_counts=True
    )

    all_values.update(values.tolist())

    for value, count in zip(values, counts):
        value_counts[int(value)] = (
            value_counts.get(int(value), 0) + int(count)
        )

    mask_count += 1

print("\n========== MASK ANALYSIS ==========")
print("Metadata records:", len(df))
print("Masks inspected:", mask_count)
print("Unique pixel values:", sorted(all_values))
print("\nTotal pixels for each value:")

for value, count in sorted(value_counts.items()):
    print(f"Value {value}: {count:,} pixels")

print("\nInspection completed.")