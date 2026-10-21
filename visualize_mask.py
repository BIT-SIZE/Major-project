from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

DATASET_PATH = Path(
    r"C:\Users\hiten\Major Project\plantseg\plantsegv2"
)

IMAGE_DIR = DATASET_PATH / "images"
MASK_DIR = DATASET_PATH / "annotations"

# Find available files
image_files = sorted([
    p for p in IMAGE_DIR.rglob("*")
    if p.is_file()
    and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
])

mask_files = sorted([
    p for p in MASK_DIR.rglob("*")
    if p.is_file()
    and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
])

print("Images found:", len(image_files))
print("Masks found:", len(mask_files))

if not image_files:
    raise FileNotFoundError("No images found in the images folder.")

if not mask_files:
    raise FileNotFoundError("No masks found in the annotations folder.")

# Display the first image to inspect the dataset
image_path = image_files[0]

image = Image.open(image_path)

print("\nSelected image:", image_path.name)
print("Image dimensions:", image.size)
print("Image mode:", image.mode)

plt.figure(figsize=(8, 6))
plt.imshow(image)
plt.title("Sample PlantSeg Image")
plt.axis("off")
plt.show()

# Display the first annotation separately for inspection
mask_path = mask_files[0]

mask = Image.open(mask_path)

mask_array = np.array(mask)

print("\nSelected annotation:", mask_path.name)
print("Mask dimensions:", mask.size)
print("Mask mode:", mask.mode)
print("Mask array shape:", mask_array.shape)
print("Mask data type:", mask_array.dtype)

print("\nUnique mask values:")

unique_values = np.unique(mask_array)

print("Number of unique values:", len(unique_values))
print("First 30 unique values:", unique_values[:30])

plt.figure(figsize=(8, 6))

if mask_array.ndim == 2:
    plt.imshow(mask_array, cmap="gray")
else:
    plt.imshow(mask_array)

plt.title("Sample PlantSeg Annotation")
plt.axis("off")
plt.show()