import json
from pathlib import Path

DATASET_PATH = Path(
    r"C:\Users\hiten\Major Project\plantseg\plantsegv2"
)

JSON_PATH = DATASET_PATH / "coco_annotations.json"

with open(JSON_PATH, "r", encoding="utf-8") as f:
    coco = json.load(f)

print("=" * 60)
print("COCO ANNOTATION ANALYSIS")
print("=" * 60)

print("\nTop-level keys:")
print(list(coco.keys()))

categories = coco.get("categories", [])

print("\nNumber of categories:", len(categories))

print("\nFirst 20 categories:")

for category in categories[:20]:
    print(category)

print("\nLast 10 categories:")

for category in categories[-10:]:
    print(category)

category_ids = [
    category["id"]
    for category in categories
]

print("\nMinimum category ID:", min(category_ids))
print("Maximum category ID:", max(category_ids))
print("Unique category IDs:", len(set(category_ids)))

print("\nFirst 5 annotations:")

for annotation in coco.get("annotations", [])[:5]:
    print(annotation)

print("\nInspection completed.")