from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


DATASET_PATH = Path(
    r"C:\Users\hiten\Major Project\plantseg\plantsegv2"
)

IMAGE_DIR = DATASET_PATH / "images"
MASK_DIR = DATASET_PATH / "annotations"
CORRECTED_DIR = DATASET_PATH / "corrected_annotations"

df = pd.read_csv(DATASET_PATH / "Metadatav2.csv")

# Inspect samples from different disease categories.
sample_df = df.groupby("Disease", sort=True).head(1).head(15)

for _, row in sample_df.iterrows():

    image_name = str(row["Name"])
    mask_name = str(row["Label file"])

    image_matches = list(IMAGE_DIR.rglob(image_name))
    mask_matches = list(MASK_DIR.rglob(mask_name))

    if len(image_matches) != 1 or len(mask_matches) != 1:
        print("\nSkipping:", image_name)
        print("Image matches:", len(image_matches))
        print("Mask matches:", len(mask_matches))
        continue

    image_path = image_matches[0]
    original_mask_path = mask_matches[0]

    relative_path = original_mask_path.relative_to(MASK_DIR)
    corrected_path = CORRECTED_DIR / relative_path

    mask_path = (
        corrected_path
        if corrected_path.exists()
        else original_mask_path
    )

    with Image.open(image_path) as image:
        image_size = image.size

    with Image.open(mask_path) as mask:
        mask_array = np.asarray(mask)

    unique_values, counts = np.unique(
        mask_array,
        return_counts=True
    )

    print("\n" + "=" * 60)
    print("Image:", image_name)
    print("Disease:", row["Disease"])
    print("Image dimensions:", image_size)
    print("Mask dimensions:", (
        mask_array.shape[1],
        mask_array.shape[0]
    ))
    print("Unique mask values:", unique_values.tolist())
    print("Pixel counts:", dict(zip(
        unique_values.tolist(),
        counts.tolist()
    )))