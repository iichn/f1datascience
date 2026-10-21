import os

from .tyre_wear import clean_laps, degradation
from .tyre_wear_spark import build_laps_spark

OUT = "data/processed/task1_iichn"

laps = build_laps_spark()
clean = clean_laps(laps)
deg = degradation(clean)

os.makedirs(OUT, exist_ok=True)
clean.to_csv(f"{OUT}/laps_clean.csv", index=False)
deg.to_csv(f"{OUT}/degradation.csv", index=False)

print(f"Гонок: {laps.GrandPrix.nunique()}, кругов всего: {len(laps)}, после чистки: {len(clean)}")
print(deg.groupby("Compound")[["DegSecPerLap", "SpeedKmhPerLap"]].median())