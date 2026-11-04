"""
Feature engineering. Every feature uses only information known BEFORE the hour starts
(calendar + weather for that hour, which in deployment comes from a weather forecast).
No feature uses `casual`, `registered`, or future rows.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MORNING_PEAK = (7, 8, 9)
EVENING_PEAK = (16, 17, 18, 19)
PEAK_HOURS = MORNING_PEAK + EVENING_PEAK
NIGHT_HOURS = (0, 1, 2, 3, 4, 5)

FEATURE_RATIONALE = {
    "hr":            "Hour of day - the single strongest driver (commute vs night).",
    "weekday":       "Day of week (0=Sun). Captures Fri/Mon effects beyond the working-day flag.",
    "workingday":    "Commuting happens on working days; changes the whole daily shape.",
    "holiday":       "Public holidays behave like weekends even on Mon-Fri.",
    "is_weekend":    "Sat/Sun indicator - leisure pattern (midday peak).",
    "mnth":          "Month - seasonal temperature/daylight cycle.",
    "season":        "Coarser seasonal signal from the source data.",
    "yr":            "Growth of the system (2012 ~ +63%). Lets the model learn the level shift.",
    "morning_peak":  "Working day AND 07-09 - morning commute window.",
    "evening_peak":  "Working day AND 16-19 - evening commute window.",
    "rush_hour":     "Either commute window on a working day - where shortages are most likely.",
    "night":         "00-05 - near-zero demand; useful window for rebalancing trucks.",
    "hr_sin/hr_cos": "Cyclical hour: tells linear models that 23:00 is next to 00:00.",
    "mnth_sin/mnth_cos": "Cyclical month: December is next to January.",
    "weather":       "1 clear, 2 mist/cloud, 3 rain/snow (4 merged into 3 - only 3 rows).",
    "temp":          "Temperature (normalised). Riding is weather-sensitive.",
    "temp_sq":       "Lets a linear model capture 'too hot' - demand flattens/falls at high temps.",
    "hum":           "Humidity - proxy for discomfort and rain.",
    "windspeed":     "Wind - discomfort on a bike.",
}

EXCLUDED_COLUMNS = {
    "casual":      "Leakage: casual + registered = cnt exactly. Unknown before the hour ends.",
    "registered":  "Leakage: same reason as casual.",
    "cnt":         "Target.",
    "instant":     "Row index; it is a disguised time trend with no causal meaning. `yr` covers the trend explicitly.",
    "dteday":      "Raw date string; decomposed into yr/mnth/weekday instead.",
    "timestamp":   "Helper for sorting/splitting, not a feature.",
    "atemp":       "'Feels-like' temperature, correlation with temp ~0.99 -> redundant; kept out to avoid collinearity.",
    "weathersit":  "Replaced by `weather` (category 4 merged into 3).",
    "hum_imputed / wind_zero / disrupted_day": "Data-quality flags for analysis, not demand drivers.",
    "temp_c / atemp_c / hum_pct / wind_unscaled": "Same information as the scaled columns, used only for readable charts/app.",
}

MODEL_FEATURES = [
    "hr", "weekday", "workingday", "holiday", "is_weekend", "mnth", "season", "yr",
    "morning_peak", "evening_peak", "rush_hour", "night",
    "hr_sin", "hr_cos", "mnth_sin", "mnth_cos",
    "weather", "temp", "temp_sq", "hum", "windspeed",
]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["is_weekend"] = out["weekday"].isin([0, 6]).astype(int)
    if "hr" in out:
        hr = out["hr"]
        out["morning_peak"] = (hr.isin(MORNING_PEAK) & (out["workingday"] == 1)).astype(int)
        out["evening_peak"] = (hr.isin(EVENING_PEAK) & (out["workingday"] == 1)).astype(int)
        out["rush_hour"] = (out["morning_peak"] | out["evening_peak"]).astype(int)
        out["night"] = hr.isin(NIGHT_HOURS).astype(int)
        out["hr_sin"] = np.sin(2 * np.pi * hr / 24)
        out["hr_cos"] = np.cos(2 * np.pi * hr / 24)
    out["mnth_sin"] = np.sin(2 * np.pi * (out["mnth"] - 1) / 12)
    out["mnth_cos"] = np.cos(2 * np.pi * (out["mnth"] - 1) / 12)
    out["temp_sq"] = out["temp"] ** 2
    out["day_type"] = np.where(out["workingday"] == 1, "Working day", "Non-working day")
    return out


def daily_peak_share(hour_df: pd.DataFrame, min_hours: int = 22) -> pd.DataFrame:
    """Per day: share of rentals in the commute windows (07-09, 16-19), for every day type.

    Uses clock-hour windows (not the working-day-only flags) so working and non-working days
    are measured on the same 7 hours. Days with < `min_hours` recorded hours are dropped,
    because a partial day distorts the share.
    """
    g = hour_df.assign(in_peak=hour_df["hr"].isin(PEAK_HOURS) * hour_df["cnt"])
    daily = g.groupby("dteday").agg(
        cnt=("cnt", "sum"), peak_cnt=("in_peak", "sum"), n_hours=("hr", "size"),
        workingday=("workingday", "first"), yr=("yr", "first"), season=("season", "first"),
        mnth=("mnth", "first"), weather_mean=("weather", "mean"), temp=("temp", "mean"),
        hum=("hum", "mean"), windspeed=("windspeed", "mean"),
    ).reset_index()
    daily = daily[daily["n_hours"] >= min_hours].copy()
    daily["peak_share"] = daily["peak_cnt"] / daily["cnt"]
    return daily
