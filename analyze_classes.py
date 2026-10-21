import pandas as pd
from pathlib import Path

DATASET_PATH = Path(
    r"C:\Users\hiten\Major Project\plantseg\plantsegv2"
)

df = pd.read_csv(DATASET_PATH / "Metadatav2.csv")

# Count images for each disease
disease_counts = (
    df.groupby("Disease")
    .size()
    .sort_values(ascending=False)
)

print("\nNumber of disease categories:", df["Disease"].nunique())

print("\nImages per disease:")
print(disease_counts.to_string())

# Count images for each plant
plant_counts = (
    df.groupby("Plant")
    .size()
    .sort_values(ascending=False)
)

print("\nNumber of plant categories:", df["Plant"].nunique())

print("\nImages per plant:")
print(plant_counts.to_string())

# Inspect dataset splits
print("\nDataset split distribution:")
print(df["Split"].value_counts(dropna=False))

# Save reports
disease_counts.to_csv(DATASET_PATH / "disease_counts.csv")
plant_counts.to_csv(DATASET_PATH / "plant_counts.csv")

print("\nReports saved successfully.")