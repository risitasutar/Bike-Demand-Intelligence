"""
Loading and cleaning for the Bike Sharing dataset.

Rules followed:
* Raw CSVs are never modified on disk; every function returns a new DataFrame.
* Nothing is deleted. Bad values are corrected and FLAGGED so the change is visible.
* Every cleaning decision is listed in CLEANING_LOG (shown in the notebook / report).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Readme: values were divided by these maxima -> multiply back for human-readable units
TEMP_MAX_C, ATEMP_MAX_C, HUM_MAX, WIND_MAX = 41, 50, 100, 67

# Days on which the system was (partly) shut down or data are incomplete (< 22 recorded hours).
# Identified in Phase 1; kept for modelling but flagged.
DISRUPTED_DAYS = pd.to_datetime([
    "2011-01-18", "2011-01-26", "2011-01-27", "2011-02-22",
    "2011-08-27", "2011-08-28", "2012-10-29", "2012-10-30",
])

CLEANING_LOG = [
    ("hum == 0 on 2011-03-10 (22 hourly rows, 1 daily row)",
     "Physically impossible -> sensor error. Replaced with the mean humidity of the same hour on "
     "the previous and next day (daily file: mean of neighbouring days). Flag column `hum_imputed`."),
    ("weathersit == 4 (3 hourly rows)",
     "Too rare to learn from. New column `weather` merges 4 into 3 ('rain/snow'). Raw column kept."),
    ("windspeed == 0 (12.5% of hours; next value is 0.0896)",
     "Likely 'calm / not recorded'. Kept unchanged, flagged in `wind_zero`. Low impact on demand."),
    ("165 missing hours (mostly 02:00-05:00; storm shutdowns)",
     "Not filled with invented rows for EDA/model training. Zero-rental hours appear to be absent, "
     "so night-time averages are slightly overstated. Storm days flagged in `disrupted_day`."),
    ("High cnt values (up to 977/hr)",
     "Genuine commute peaks (Phase 1, Step 10). Not removed."),
]


def load_hour(path: str = "data/hour.csv") -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["dteday"])


def load_day(path: str = "data/day.csv") -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["dteday"])


def _impute_hourly_humidity(df: pd.DataFrame) -> pd.DataFrame:
    bad = df["hum"] == 0
    df["hum_imputed"] = bad.astype(int)
    if not bad.any():
        return df
    lookup = df.loc[~bad].set_index(["dteday", "hr"])["hum"]
    for idx in df.index[bad]:
        d, h = df.at[idx, "dteday"], df.at[idx, "hr"]
        neighbours = [lookup.get((d + pd.Timedelta(days=k), h)) for k in (-1, 1)]
        neighbours = [v for v in neighbours if v is not None and not pd.isna(v)]
        df.at[idx, "hum"] = np.mean(neighbours) if neighbours else df.loc[~bad, "hum"].median()
    return df


def clean_hour(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy().sort_values(["dteday", "hr"]).reset_index(drop=True)
    df["timestamp"] = df["dteday"] + pd.to_timedelta(df["hr"], unit="h")
    df = _impute_hourly_humidity(df)
    df["weather"] = df["weathersit"].clip(upper=3)
    df["wind_zero"] = (df["windspeed"] == 0).astype(int)
    df["disrupted_day"] = df["dteday"].isin(DISRUPTED_DAYS).astype(int)
    return add_physical_units(df)


def clean_day(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy().sort_values("dteday").reset_index(drop=True)
    bad = df["hum"] == 0
    df["hum_imputed"] = bad.astype(int)
    for idx in df.index[bad]:
        neigh = df.loc[[i for i in (idx - 1, idx + 1) if i in df.index and not bad[i]], "hum"]
        df.at[idx, "hum"] = neigh.mean()
    df["weather"] = df["weathersit"].clip(upper=3)
    df["disrupted_day"] = df["dteday"].isin(DISRUPTED_DAYS).astype(int)
    return add_physical_units(df)


def add_physical_units(df: pd.DataFrame) -> pd.DataFrame:
    """De-normalise for readable charts and for the prototype (no new information)."""
    df["temp_c"] = df["temp"] * TEMP_MAX_C
    df["atemp_c"] = df["atemp"] * ATEMP_MAX_C
    df["hum_pct"] = df["hum"] * HUM_MAX
    df["wind_unscaled"] = df["windspeed"] * WIND_MAX   # Readme does not state the unit
    return df
