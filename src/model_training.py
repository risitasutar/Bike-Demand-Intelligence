"""
Model training and evaluation for hourly bike demand (target = cnt).

Design rules
* Chronological evaluation only: train <= 2012-06-30, test = 2012-07-01 .. 2012-12-31.
* Model selection uses expanding-window CV inside the training period (monthly folds Mar-Jun 2012).
  The test set is used once, for the chosen model.
* Inputs = feature_engineering.MODEL_FEATURES (no casual/registered/instant).
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

import feature_engineering as fe

SEED = 42
TRAIN_END = pd.Timestamp("2012-06-30")
TEST_START = pd.Timestamp("2012-07-01")
CV_MONTHS = ["2012-03", "2012-04", "2012-05", "2012-06"]   # expanding-window validation months
LAG_FEATURES = ["lag_168", "same_hour_4wk_mean"]


# ------------------------------------------------------------------ splitting
def chrono_split(df: pd.DataFrame):
    train = df[df["dteday"] <= TRAIN_END].copy()
    test = df[df["dteday"] >= TEST_START].copy()
    return train, test


def time_folds(train: pd.DataFrame):
    """Yield (fold_name, train_idx, valid_idx): fit on everything before the month, validate on the month."""
    for m in CV_MONTHS:
        start = pd.Timestamp(m + "-01")
        end = start + pd.offsets.MonthEnd(0)
        tr = train.index[train["dteday"] < start]
        va = train.index[(train["dteday"] >= start) & (train["dteday"] <= end)]
        yield m, tr, va


# ------------------------------------------------------------------ baselines
class HourProfileBaseline(BaseEstimator, RegressorMixin):
    """'What an ops manager would do without a model': average demand for this hour and day type,
    from the training data only."""

    def fit(self, X, y):
        d = pd.DataFrame({"hr": X["hr"].values, "wd": X["workingday"].values, "y": np.asarray(y)})
        self.table_ = d.groupby(["hr", "wd"]).y.mean()
        self.global_ = float(np.mean(y))
        return self

    def predict(self, X):
        keys = list(zip(X["hr"].values, X["workingday"].values))
        return np.array([self.table_.get(k, self.global_) for k in keys])


# ------------------------------------------------------------------ linear model features
def add_hour_daytype(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    X["hr_wd"] = X["hr"].astype(int) * 10 + X["workingday"].astype(int)   # 48 hour x day-type cells
    return X


LINEAR_CATEGORICAL = ["hr_wd", "weather", "mnth"]
LINEAR_NUMERIC = ["yr", "holiday", "temp", "temp_sq", "hum", "windspeed"]
# not used by the linear model: is_weekend (collinear with workingday+holiday), season (redundant with mnth),
# hr/weekday as numbers, peak flags and sin/cos (all absorbed by the 48 hour x day-type dummies)


def linear_pipeline() -> Pipeline:
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), LINEAR_CATEGORICAL),
        ("num", StandardScaler(), LINEAR_NUMERIC),
    ])
    return Pipeline([("hour_x_daytype", FunctionTransformer(add_hour_daytype)), ("prep", pre),
                     ("model", LinearRegression())])


def log_target(model):
    return TransformedTargetRegressor(regressor=model, func=np.log1p, inverse_func=np.expm1)


def candidate_models() -> dict:
    """Name -> unfitted estimator. Kept deliberately small: one or two settings per family."""
    rf = lambda leaf: RandomForestRegressor(n_estimators=300, min_samples_leaf=leaf, max_features=0.5,
                                            n_jobs=-1, random_state=SEED)
    gb = lambda depth: GradientBoostingRegressor(n_estimators=400, learning_rate=0.05, max_depth=depth,
                                                 subsample=0.8, random_state=SEED)
    return {
        "Dummy (training mean)": DummyRegressor(strategy="mean"),
        "Baseline: hour x day-type average": HourProfileBaseline(),
        "Linear regression": linear_pipeline(),
        "Linear regression (log target)": log_target(linear_pipeline()),
        "Decision tree (depth 10)": DecisionTreeRegressor(max_depth=10, min_samples_leaf=5, random_state=SEED),
        "Decision tree (depth 14)": DecisionTreeRegressor(max_depth=14, min_samples_leaf=5, random_state=SEED),
        "Random forest (leaf 1)": rf(1),
        "Random forest (leaf 3)": rf(3),
        "Random forest (leaf 3, log target)": log_target(rf(3)),
        "Gradient boosting (depth 4)": gb(4),
        "Gradient boosting (depth 6)": gb(6),
        "Gradient boosting (depth 6, log target)": log_target(gb(6)),
    }


# ------------------------------------------------------------------ metrics
def evaluate(y_true, y_pred, is_peak=None) -> dict:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    err = y_pred - y_true
    out = {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
        # WAPE instead of MAPE: MAPE explodes on hours with 1-5 rentals
        "WAPE_%": 100 * np.abs(err).sum() / y_true.sum(),
        "Bias (mean pred - actual)": err.mean(),
    }
    if is_peak is not None:
        p = np.asarray(is_peak, bool)
        out["Peak MAE"] = np.abs(err[p]).mean()
        out["Peak bias"] = err[p].mean()
        out["Peak under-predicted %"] = 100 * (err[p] < 0).mean()
    return out


def cross_validate(models: dict, train: pd.DataFrame, features: list, target: str = "cnt") -> pd.DataFrame:
    rows = []
    for name, est in models.items():
        t0 = time.time()
        for fold, tr, va in time_folds(train):
            m = clone(est).fit(train.loc[tr, features], train.loc[tr, target])
            pred = np.clip(m.predict(train.loc[va, features]), 0, None)
            rows.append({"model": name, "fold": fold, **evaluate(train.loc[va, target], pred, train.loc[va, "rush_hour"])})
        rows[-1]["fit_seconds"] = time.time() - t0
    res = pd.DataFrame(rows)
    summary = res.groupby("model", sort=False).mean(numeric_only=True)
    return summary.sort_values("MAE"), res


# ------------------------------------------------------------------ lag features (variant)
def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Same hour last week, and mean of the same hour over the previous 4 weeks.

    Both are known 7 days in advance, so they are valid for forecasts up to a week ahead.
    Missing hours are treated as 0 rentals (Phase 1: absent hours are almost all 02:00-05:00 / storm closures).
    """
    s = df.set_index("timestamp")["cnt"]
    full = pd.date_range(s.index.min(), s.index.max(), freq="h")
    s = s.reindex(full).fillna(0)
    lags = pd.concat({k: s.shift(168 * k) for k in (1, 2, 3, 4)}, axis=1)
    feats = pd.DataFrame({"lag_168": lags[1], "same_hour_4wk_mean": lags.mean(axis=1, skipna=False)})
    out = df.join(feats, on="timestamp")
    return out


# ------------------------------------------------------------------ prediction intervals
def quantile_models(alpha_low=0.1, alpha_high=0.9) -> dict:
    q = lambda a: GradientBoostingRegressor(loss="quantile", alpha=a, n_estimators=400, learning_rate=0.05,
                                            max_depth=6, subsample=0.8, random_state=SEED)
    return {f"P{int(alpha_low * 100)}": q(alpha_low), f"P{int(alpha_high * 100)}": q(alpha_high)}


def out_of_fold_predictions(est, train: pd.DataFrame, features: list, target: str = "cnt") -> pd.Series:
    preds = []
    for _, tr, va in time_folds(train):
        m = clone(est).fit(train.loc[tr, features], train.loc[tr, target])
        preds.append(pd.Series(np.clip(m.predict(train.loc[va, features]), 0, None), index=va))
    return pd.concat(preds)


def planning_multipliers(actual: pd.Series, predicted: pd.Series, is_rush: pd.Series, q: float = 0.9) -> dict:
    """Empirical q-quantile of (actual+1)/(predicted+1) on validation folds, separately for rush / other hours.
    Planning forecast = (point + 1) * multiplier - 1  ->  covers ~q of hours if errors stay similar."""
    ratio = (np.asarray(actual) + 1) / (np.asarray(predicted) + 1)
    grp = np.where(np.asarray(is_rush) == 1, "rush", "other")
    return pd.Series(ratio, index=grp).groupby(level=0).quantile(q).to_dict()


def planning_forecast(point: np.ndarray, is_rush, multipliers: dict) -> np.ndarray:
    f = np.where(np.asarray(is_rush) == 1, multipliers["rush"], multipliers["other"])
    return np.clip((np.asarray(point) + 1) * f - 1, 0, None)


def walk_forward(est, df: pd.DataFrame, features: list, months: list, target: str = "cnt") -> pd.Series:
    """Retrain on all data before each month, predict that month (simulates monthly retraining)."""
    preds = []
    for m in months:
        start = pd.Timestamp(m + "-01")
        end = start + pd.offsets.MonthEnd(0)
        a = df[df["dteday"] < start]
        b = df[(df["dteday"] >= start) & (df["dteday"] <= end)]
        mm = clone(est).fit(a[features], a[target])
        preds.append(pd.Series(np.clip(mm.predict(b[features]), 0, None), index=b.index))
    return pd.concat(preds)


def bikes_in_use(rentals_per_hour, trip_minutes: float):
    """Little's law: average number of bikes on the road = arrival rate x average trip time."""
    return rentals_per_hour * trip_minutes / 60          # works for numbers, arrays, Series and DataFrames


def save_bundle(path: str, **items):
    import joblib, sklearn
    items["sklearn_version"] = sklearn.__version__
    joblib.dump(items, path)


def load_bundle(path: str) -> dict:
    import joblib
    return joblib.load(path)


# ------------------------------------------------------------------ scenarios (used by the prototype)
def season_for(date) -> int:
    """Season coding used in the dataset (verified on day.csv): changes on 21 Mar, 21 Jun, 23 Sep, 21 Dec."""
    md = (pd.Timestamp(date).month, pd.Timestamp(date).day)
    if md >= (12, 21) or md < (3, 21):
        return 1
    if md < (6, 21):
        return 2
    if md < (9, 23):
        return 3
    return 4


def scenario_frame(date, hours=range(24), weather: int = 1, temp_c: float | None = None,
                   hum_pct: float | None = None, wind: float | None = None, holiday: int = 0,
                   climate: pd.DataFrame | None = None, yr: int = 1) -> pd.DataFrame:
    """Build model-ready rows for a date/hours under a weather scenario.

    Missing weather values default to the historical median for that month and hour (`climate`).
    `yr` is capped at 1 (2012 demand level): the model has never seen a later year.
    """
    date = pd.Timestamp(date)
    rows = []
    for h in hours:
        clim = climate.loc[(date.month, h)] if climate is not None else None
        t = temp_c if temp_c is not None else clim["temp_c"]
        hu = hum_pct if hum_pct is not None else clim["hum_pct"]
        w = wind if wind is not None else clim["wind_unscaled"]
        weekday = (date.dayofweek + 1) % 7                     # pandas Mon=0 -> dataset Sun=0
        workingday = int(weekday in (1, 2, 3, 4, 5) and not holiday)
        rows.append({"dteday": date.normalize(), "hr": h, "weekday": weekday, "workingday": workingday,
                     "holiday": int(holiday), "mnth": date.month, "season": season_for(date), "yr": min(int(yr), 1),
                     "weather": int(weather), "temp": t / 41, "hum": hu / 100, "windspeed": w / 67})
    return fe.add_features(pd.DataFrame(rows))


def climate_table(hour_df: pd.DataFrame) -> pd.DataFrame:
    """Median temperature / humidity / wind by (month, hour) - default weather for scenarios."""
    return hour_df.groupby(["mnth", "hr"])[["temp_c", "hum_pct", "wind_unscaled"]].median()
