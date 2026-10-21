from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

DATASET_PATH = Path(
    r"C:\Users\hiten\Major Project\plantseg\plantsegv2"
)

IMAGE_DIR = DATASET_PATH / "images"
MASK_DIR = DATASET_PATH / "annotations"

df = pd.read_csv(DATASET_PATH / "Metadatav2.csv")

# Select one metadata record
row = df.iloc[0]

image_path = IMAGE_DIR / str(row["Name"])
mask_path = MASK_DIR / str(row["Label file"])

# Load the image and mask
image = Image.open(image_path).convert("RGB")
mask = Image.open(mask_path).convert("L")

image_array = np.array(image)
mask_array = np.array(mask)

print("Image:", row["Name"])
print("Mask:", row["Label file"])
print("Plant:", row["Plant"])
print("Disease:", row["Disease"])
print("Split:", row["Split"])

print("\nImage shape:", image_array.shape)
print("Mask shape:", mask_array.shape)
print("Mask values:", np.unique(mask_array))

# Ensure the dimensions match before overlaying
if image_array.shape[:2] != mask_array.shape:
    raise ValueError(
        "Image and mask dimensions do not match."
    )

# Display image, mask, and overlay
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

axes[0].imshow(image_array)
axes[0].set_title("Original Image")

axes[1].imshow(mask_array, cmap="gray", vmin=0, vmax=1)
axes[1].set_title("Disease Annotation")

axes[2].imshow(image_array)
axes[2].imshow(
    mask_array,
    cmap="Reds",
    alpha=0.45,
    vmin=0,
    vmax=1
)
axes[2].set_title("Image + Mask Overlay")

for ax in axes:
    ax.axis("off")

plt.tight_layout()
plt.show()