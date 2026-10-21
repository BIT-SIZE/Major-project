from pathlib import Path
import pandas as pd
from PIL import Image

DATASET_PATH = Path(
    r"C:\Users\hiten\Major Project\plantseg\plantsegv2"
)

IMAGE_DIR = DATASET_PATH / "images"
MASK_DIR = DATASET_PATH / "annotations"
CORRECTED_DIR = DATASET_PATH / "corrected_annotations"

df = pd.read_csv(DATASET_PATH / "Metadatav2.csv")


def find_file(root, filename):
    matches = list(root.rglob(filename))

    if len(matches) != 1:
        raise ValueError(
            f"Expected one file for {filename}, "
            f"found {len(matches)}"
        )

    return matches[0]


remaining_mismatches = []

for _, row in df.iterrows():

    image_path = find_file(
        IMAGE_DIR,
        str(row["Name"])
    )

    original_mask_path = find_file(
        MASK_DIR,
        str(row["Label file"])
    )

    # Use the corrected copy when one exists.
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
        mask_size = mask.size

    if image_size != mask_size:
        remaining_mismatches.append(
            {
                "image": str(row["Name"]),
                "mask": str(row["Label file"]),
                "image_size": image_size,
                "mask_size": mask_size,
            }
        )

print("Remaining dimension mismatches:", len(remaining_mismatches))

for item in remaining_mismatches:
    print(item)