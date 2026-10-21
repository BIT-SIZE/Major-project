from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

DATASET_PATH = Path(
    r"C:\Users\hiten\Major Project\plantseg\plantsegv2"
)

IMAGE_DIR = DATASET_PATH / "images"
MASK_DIR = DATASET_PATH / "annotations"

image_name = "apple_rust_google_0065.jpg"
mask_name = "apple_rust_google_0065.png"

image_matches = list(IMAGE_DIR.rglob(image_name))
mask_matches = list(MASK_DIR.rglob(mask_name))

if len(image_matches) != 1 or len(mask_matches) != 1:
    raise ValueError(
        "Expected exactly one matching image and mask."
    )

image_path = image_matches[0]
mask_path = mask_matches[0]

image = Image.open(image_path).convert("RGB")
mask = Image.open(mask_path).convert("L")

image_array = np.array(image)
mask_array = np.array(mask)

print("Image dimensions:", image.size)
print("Mask dimensions:", mask.size)
print("Mask unique values:", np.unique(mask_array))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].imshow(image_array)
axes[0].set_title("Original Image")
axes[0].axis("off")

axes[1].imshow(mask_array, cmap="gray")
axes[1].set_title("Original Annotation")
axes[1].axis("off")

plt.tight_layout()
plt.show()