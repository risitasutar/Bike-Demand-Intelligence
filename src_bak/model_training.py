"""
Model training / evaluation for hourly bike demand.

Rules followed
* Chronological split only (random splits leak neighbouring hours into training).
* No casual / registered / instant in features (see feature_engineering.EXCLUDED_COLUMNS).
* Hyper-parameters are chosen with expanding-window (time-ordered) CV inside the TRAIN period only;
  the test period is touched once, for the final comparison.
"""
from __future__ import annotations

import time

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from feature_engineering import MODEL_FEATURES, PEAK_HOURS

SEED = 42
TEST_START = pd.Timestamp("2012-07-01")   # train: Jan-2011..Jun-2012, test: Jul..Dec-2012


def chrono_split(df: pd.DataFrame, test_start=TEST_START):
    df = df.sort_values("timestamp")
    return df[df.timestamp < test_start].copy(), df[df.timestamp >= test_start].copy()


def make_models() -> dict:
    return {
        "Baseline (mean)": DummyRegressor(strategy="mean"),
        "Linear regression": make_pipeline(StandardScaler(), LinearRegression()),
        "Decision tree": DecisionTreeRegressor(max_depth=10, min_samples_leaf=10, random_state=SEED),
        "Random forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=3, n_jobs=-1, random_state=SEED),
        "Gradient boosting": GradientBoostingRegressor(n_estimators=400, learning_rate=0.05, max_depth=4,
                                                       subsample=0.8, random_state=SEED),
    }


def metrics(y, p) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    return {"MAE": mean_absolute_error(y, p),
            "RMSE": float(np.sqrt(mean_squared_error(y, p))),
            "R2": r2_score(y, p),
            "Bias (mean pred-actual)": float(np.mean(p - y)),
            "WAPE %": float(100 * np.abs(p - y).sum() / y.sum())}   # robust % error (no divide-by-small-y)


def cv_scores(model, X, y, n_splits=4) -> dict:
    maes = []
    for tr, va in TimeSeriesSplit(n_splits=n_splits).split(X):
        model.fit(X.iloc[tr], y.iloc[tr])
        maes.append(mean_absolute_error(y.iloc[va], model.predict(X.iloc[va])))
    return {"CV MAE mean": float(np.mean(maes)), "CV MAE std": float(np.std(maes))}


def fit_predict(model, Xtr, ytr, Xte, log_target=False):
    t = time.time()
    model.fit(Xtr, np.log1p(ytr) if log_target else ytr)
    p = model.predict(Xte)
    p = np.expm1(p) if log_target else p
    return np.clip(p, 0, None), time.time() - t


def add_lag_week(df: pd.DataFrame) -> pd.DataFrame:
    """Demand at the same clock hour exactly 7 days earlier (merge on timestamp, NOT row shift,
    because some hours are missing). Known in advance for any forecast <= 7 days ahead."""
    lag = df[["timestamp", "cnt"]].copy()
    lag["timestamp"] = lag["timestamp"] + pd.Timedelta(days=7)
    lag = lag.rename(columns={"cnt": "lag_168"})
    return df.merge(lag, on="timestamp", how="left")


def peak_mask(df: pd.DataFrame):
    return (df.workingday == 1) & df.hr.isin(PEAK_HOURS)


def calibrate_buffer(model_factory, train: pd.DataFrame, features, q=0.90, holdout_days=92):
    """Ratio-quantile 'safety buffer': fit on train minus its last ~3 months, measure
    actual/predicted on those months, take the q-quantile. Uses only training-period data."""
    cut = train.timestamp.max().normalize() - pd.Timedelta(days=holdout_days)
    a, b = train[train.timestamp < cut], train[train.timestamp >= cut]
    m = model_factory()
    m.fit(a[features], a.cnt)
    pred = np.clip(m.predict(b[features]), 1, None)
    return float(np.quantile(b.cnt / pred, q))


def save_model(model, features, path="models/best_model.joblib", extra=None):
    joblib.dump({"model": model, "features": list(features), **(extra or {})}, path)
