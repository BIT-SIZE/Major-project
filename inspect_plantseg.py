import os

DATASET_PATH = r"C:\Users\hiten\Major Project\plantseg\plantsegv2"

print("\n========== PLANTSEG DATASET STRUCTURE ==========\n")

for root, dirs, files in os.walk(DATASET_PATH):
    level = root.replace(DATASET_PATH, "").count(os.sep)

    indent = " " * 4 * level
    print(f"{indent}{os.path.basename(root)}/")

    sub_indent = " " * 4 * (level + 1)

    for file in files[:5]:
        print(f"{sub_indent}{file}")

    if len(files) > 5:
        print(f"{sub_indent}... and {len(files) - 5} more files")