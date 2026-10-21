import json
from pathlib import Path

DATASET_PATH = Path(
    r"C:\Users\hiten\Major Project\plantseg\plantsegv2"
)

JSON_PATH = DATASET_PATH / "coco_annotations.json"

with open(JSON_PATH, "r", encoding="utf-8") as f:
    coco = json.load(f)

target_name = "apple_black_rot_1.jpg"

print("=" * 60)
print("SEARCHING COCO IMAGE FILENAMES")
print("=" * 60)

# Search for an exact filename match
exact_matches = [
    image
    for image in coco["images"]
    if image.get("file_name") == target_name
]

print("\nExact matches:", len(exact_matches))

for image in exact_matches:
    print(image)

# Search for filenames containing the distinctive stem
stem = "apple_black_rot_1"

partial_matches = [
    image
    for image in coco["images"]
    if stem in Path(image.get("file_name", "")).stem
]

print("\nPartial matches:", len(partial_matches))

for image in partial_matches[:20]:
    print(image)

# Print sample COCO filenames
print("\nFirst 20 COCO image filenames:")

for image in coco["images"][:20]:
    print(image.get("file_name"))

print("\nInspection completed.")