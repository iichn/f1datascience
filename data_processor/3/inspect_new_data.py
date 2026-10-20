import pandas as pd
import os

# We are pointing to the new Monza Race file
file_path = '../../samples/sample_2022_Round_16_Italian_Grand_Prix_R_telemetry.csv'

print("Reading the new Big Data sample...")
df = pd.read_csv(file_path)

print("=== TOTAL ROWS AND COLUMNS ===")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
print("=" * 30)

print("=== ALL COLUMN NAMES ===")
for col in df.columns:
    print(f"- {col}")

print("=" * 30)
print("=== FIRST 3 ROWS ===")
print(df.head(3))