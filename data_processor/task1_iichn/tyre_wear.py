import glob
import numpy as np
import pandas as pd

COLS = ["Year", "GrandPrix", "Driver", "Team", "LapNumber", "LapTime_Sec", "Compound",
        "TyreLife", "Stint", "TrackStatus", "Speed", "Throttle", "TrackTemp"]
KEY = ["Year", "GrandPrix", "Driver", "LapNumber"]
STINT = ["Year", "GrandPrix", "Driver", "Stint"]


def load_raw(pattern="data/raw_telemetry/*.csv"):
    return pd.concat((pd.read_csv(f, usecols=COLS) for f in glob.glob(pattern)), ignore_index=True)


def build_laps(raw):
    laps = raw.groupby(KEY, as_index=False).agg(
        Team=("Team", "first"), Compound=("Compound", "first"), TyreLife=("TyreLife", "first"),
        Stint=("Stint", "first"), LapTime=("LapTime_Sec", "first"),
        AvgSpeed=("Speed", "mean"), MaxSpeed=("Speed", "max"),
        FullThrottle=("Throttle", lambda s: (s >= 99).mean()),
        TrackTemp=("TrackTemp", "first"),
        Green=("TrackStatus", lambda s: pd.to_numeric(s, errors="coerce").eq(1).all()),
    )
    first = laps.groupby(STINT).LapNumber.transform("min")
    last = laps.groupby(STINT).LapNumber.transform("max")
    laps["InOut"] = (laps.LapNumber == first) | (laps.LapNumber == last)
    return laps


def clean_laps(laps):
    d = laps[laps.Compound.isin(["SOFT", "MEDIUM", "HARD"]) & laps.Green
             & ~laps.InOut & (laps.LapNumber > 1) & laps.LapTime.notna()].copy()
    med = d.groupby(["GrandPrix", "Driver"]).LapTime.transform("median")
    return d[d.LapTime <= 1.07 * med]


def degradation(clean, fuel_k=0.055, min_laps=6):
    d = clean.copy()
    d["LapTimeFC"] = d.LapTime + fuel_k * (d.LapNumber - 1)
    d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=["TyreLife", "LapTimeFC", "AvgSpeed"])
    rows = []
    for (gp, drv, stint, comp), s in d.groupby(["GrandPrix", "Driver", "Stint", "Compound"]):
        if len(s) < min_laps:
            continue
        rows.append(dict(GrandPrix=gp, Driver=drv, Stint=stint, Compound=comp, Laps=len(s),
                         DegSecPerLap=np.polyfit(s.TyreLife, s.LapTimeFC, 1)[0],
                         SpeedKmhPerLap=np.polyfit(s.TyreLife, s.AvgSpeed, 1)[0]))
    return pd.DataFrame(rows)