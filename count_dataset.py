from pathlib import Path

DATASET_PATH = Path(
    r"C:\Users\hiten\Major Project\plantseg\plantsegv2"
)

IMAGE_DIR = DATASET_PATH / "images"
MASK_DIR = DATASET_PATH / "annotations"

image_extensions = {".jpg", ".jpeg", ".png", ".webp"}

images = [
    f for f in IMAGE_DIR.rglob("*")
    if f.is_file() and f.suffix.lower() in image_extensions
]

masks = [
    f for f in MASK_DIR.rglob("*")
    if f.is_file() and f.suffix.lower() in image_extensions
]

print("Total images:", len(images))
print("Total annotation files:", len(masks))

print("\nSample image filenames:")

for image in images[:10]:
    print(image.name)

print("\nSample annotation filenames:")

for mask in masks[:10]:
    print(mask.name)