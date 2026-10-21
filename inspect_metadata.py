import pandas as pd

CSV_PATH = (
    r"C:\Users\hiten\Major Project"
    r"\plantseg\plantsegv2\Metadatav2.csv"
)

df = pd.read_csv(CSV_PATH)

print("\n========== METADATA INFORMATION ==========\n")

print("Dataset shape:", df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nFirst 10 rows:")
print(df.head(10).to_string())

print("\nMissing values:")
print(df.isnull().sum())

print("\nData types:")
print(df.dtypes)