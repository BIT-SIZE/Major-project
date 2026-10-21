import json

JSON_PATH = (
    r"C:\Users\hiten\Major Project"
    r"\plantseg\plantsegv2\coco_annotations.json"
)

with open(JSON_PATH, "r", encoding="utf-8") as file:
    data = json.load(file)

print("Top-level keys:")
print(list(data.keys()))

print("\nNumber of images:", len(data.get("images", [])))
print("Number of annotations:", len(data.get("annotations", [])))
print("Number of categories:", len(data.get("categories", [])))

print("\nFirst image record:")

if data.get("images"):
    print(data["images"][0])

print("\nFirst annotation record:")

if data.get("annotations"):
    print(data["annotations"][0])

print("\nCategories:")

for category in data.get("categories", [])[:20]:
    print(category)