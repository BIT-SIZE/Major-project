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

image_path = next(IMAGE_DIR.rglob(image_name))
mask_path = next(MASK_DIR.rglob(mask_name))

image = Image.open(image_path).convert("RGB")
mask = Image.open(mask_path)

image_array = np.array(image)
mask_array = np.array(mask)

print("Original image size:", image.size)
print("Original mask size:", mask.size)
print("Original mask values:", np.unique(mask_array))

# Try the four possible orientations without resizing.
candidates = {
    "Original": mask,
    "Rotate 90 degrees clockwise": mask.rotate(-90, expand=True),
    "Rotate 90 degrees counterclockwise": mask.rotate(90, expand=True),
    "Rotate 180 degrees": mask.rotate(180, expand=True),
}

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

for ax, (title, candidate) in zip(axes.flat, candidates.items()):
    candidate_array = np.array(candidate)

    ax.imshow(image_array)
    ax.imshow(
        candidate_array == 3,
        cmap="Reds",
        alpha=0.5,
        vmin=0,
        vmax=1
    )

    ax.set_title(title)
    ax.axis("off")

    print(title, "size:", candidate.size)

plt.tight_layout()
plt.show()