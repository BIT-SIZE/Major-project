from pathlib import Path
import pandas as pd
from PIL import Image

DATASET_PATH = Path(
    r"C:\Users\hiten\Major Project\plantseg\plantsegv2"
)

IMAGE_DIR = DATASET_PATH / "images"
MASK_DIR = DATASET_PATH / "annotations"
CSV_PATH = DATASET_PATH / "Metadatav2.csv"

df = pd.read_csv(CSV_PATH)


def find_file(root, filename):
    matches = list(root.rglob(filename))

    if len(matches) == 1:
        return matches[0]

    if len(matches) == 0:
        return None

    raise ValueError(
        f"Multiple files found for {filename}: {matches}"
    )


print("=" * 70)
print("INSPECTING IMAGE-MASK DIMENSION MISMATCHES")
print("=" * 70)

mismatch_count = 0

for _, row in df.iterrows():

    image_name = str(row["Name"])
    mask_name = str(row["Label file"])

    image_path = find_file(IMAGE_DIR, image_name)
    mask_path = find_file(MASK_DIR, mask_name)

    if image_path is None or mask_path is None:
        continue

    with Image.open(image_path) as image:
        image_size = image.size
        image_mode = image.mode

    with Image.open(mask_path) as mask:
        mask_size = mask.size
        mask_mode = mask.mode

    if image_size != mask_size:
        mismatch_count += 1

        print(f"\nMismatch #{mismatch_count}")
        print("Image:", image_name)
        print("Mask:", mask_name)
        print("Image path:", image_path)
        print("Mask path:", mask_path)
        print("Image dimensions:", image_size)
        print("Mask dimensions:", mask_size)
        print("Image mode:", image_mode)
        print("Mask mode:", mask_mode)

print("\nTotal mismatches:", mismatch_count)