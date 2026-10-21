from pathlib import Path
import pandas as pd
from PIL import Image

DATASET_PATH = Path(
    r"C:\Users\hiten\Major Project\plantseg\plantsegv2"
)

IMAGE_DIR = DATASET_PATH / "images"
MASK_DIR = DATASET_PATH / "annotations"

OUTPUT_DIR = DATASET_PATH / "corrected_annotations"

CSV_PATH = DATASET_PATH / "Metadatav2.csv"

df = pd.read_csv(CSV_PATH)


def find_file(root, filename):
    matches = list(root.rglob(filename))

    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one file for {filename}, "
            f"but found {len(matches)}"
        )

    return matches[0]


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

corrected_count = 0

for _, row in df.iterrows():

    image_name = str(row["Name"])
    mask_name = str(row["Label file"])

    image_path = find_file(IMAGE_DIR, image_name)
    mask_path = find_file(MASK_DIR, mask_name)

    with Image.open(image_path) as image:
        image_size = image.size

    with Image.open(mask_path) as mask:
        original_mask = mask.copy()

    if image_size == original_mask.size:
        continue

    # Only proceed if the dimensions are exactly reversed.
    if image_size != (
        original_mask.height,
        original_mask.width
    ):
        print("Skipping unexpected mismatch:", mask_name)
        continue

    # Rotate the mask 90 degrees clockwise.
    corrected_mask = original_mask.rotate(
        -90,
        expand=True
    )

    if corrected_mask.size != image_size:
        print("Correction failed:", mask_name)
        continue

    # Preserve the original folder structure.
    relative_path = mask_path.relative_to(MASK_DIR)
    output_path = OUTPUT_DIR / relative_path

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    corrected_mask.save(output_path)

    corrected_count += 1

    print(
        f"Corrected: {mask_name} | "
        f"Original: {original_mask.size} | "
        f"Corrected: {corrected_mask.size}"
    )

print("\nTotal corrected masks:", corrected_count)
print("Corrected masks saved in:", OUTPUT_DIR)
print("Original annotations were not modified.")