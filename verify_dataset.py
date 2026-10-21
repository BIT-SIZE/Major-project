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

print("=" * 60)
print("PLANTSEG DATASET VERIFICATION")
print("=" * 60)


def find_file(root, filename):
    """
    Find a file inside a directory and its subdirectories.
    """
    matches = list(root.rglob(filename))

    if not matches:
        return None

    if len(matches) > 1:
        print(f"Warning: Multiple files found for {filename}")

    return matches[0]


missing_images = []
missing_masks = []
dimension_mismatches = []

for _, row in df.iterrows():

    image_name = str(row["Name"])
    mask_name = str(row["Label file"])

    image_path = find_file(IMAGE_DIR, image_name)
    mask_path = find_file(MASK_DIR, mask_name)

    if image_path is None:
        missing_images.append(image_name)
        continue

    if mask_path is None:
        missing_masks.append(mask_name)
        continue

    with Image.open(image_path) as image:
        image_size = image.size

    with Image.open(mask_path) as mask:
        mask_size = mask.size

    if image_size != mask_size:
        dimension_mismatches.append(
            {
                "image": image_name,
                "mask": mask_name,
                "image_size": image_size,
                "mask_size": mask_size,
            }
        )

print("\nMetadata records:", len(df))
print("Missing images:", len(missing_images))
print("Missing masks:", len(missing_masks))
print(
    "Image-mask dimension mismatches:",
    len(dimension_mismatches)
)

if missing_images:
    print("\nFirst missing images:")
    print(missing_images[:10])

if missing_masks:
    print("\nFirst missing masks:")
    print(missing_masks[:10])

if dimension_mismatches:
    print("\nFirst dimension mismatches:")
    print(dimension_mismatches[:10])

print("\nVerification completed.")