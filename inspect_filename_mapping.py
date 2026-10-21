import json
from pathlib import Path

import pandas as pd
from PIL import Image


DATASET_PATH = Path(
    r"C:\Users\hiten\Major Project\plantseg\plantsegv2"
)

CSV_PATH = DATASET_PATH / "Metadatav2.csv"
JSON_PATH = DATASET_PATH / "coco_annotations.json"
IMAGE_DIR = DATASET_PATH / "images"


# Load metadata
df = pd.read_csv(CSV_PATH)

# Load COCO JSON
with open(JSON_PATH, "r", encoding="utf-8") as f:
    coco = json.load(f)


# Index COCO images by filename
coco_by_name = {
    item["file_name"]: item
    for item in coco["images"]
}


print("=" * 70)
print("METADATA AND COCO FILENAME COMPARISON")
print("=" * 70)


for _, row in df.head(20).iterrows():

    filename = str(row["Name"])

    matches = list(IMAGE_DIR.rglob(filename))

    if len(matches) != 1:
        print(f"\n{filename}")
        print("Image file matches:", len(matches))
        continue

    image_path = matches[0]

    with Image.open(image_path) as image:
        actual_width, actual_height = image.size

    print(f"\nMetadata filename: {filename}")
    print(
        "Actual dimensions:",
        actual_width,
        "x",
        actual_height
    )

    coco_record = coco_by_name.get(filename)

    if coco_record is not None:

        print("COCO filename: EXACT MATCH")
        print("COCO image ID:", coco_record["id"])
        print(
            "COCO dimensions:",
            coco_record["width"],
            "x",
            coco_record["height"]
        )

    else:

        print("COCO filename: NO EXACT MATCH")

        # Display candidates only; do not treat them as matches.
        stem = Path(filename).stem

        candidates = [
            item
            for item in coco["images"]
            if Path(item["file_name"]).stem == stem
        ]

        print("Same-stem candidates:", len(candidates))

        for candidate in candidates[:5]:
            print("  ", candidate)


print("\nInspection completed.")
