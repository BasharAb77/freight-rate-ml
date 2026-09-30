"""Freight-rate model: cleaning, features, final fit and predictions."""
from __future__ import annotations

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold

EQUIP = {"Dry Van": 0, "Reefer": 1, "Flatbed": 2}
STATIC = ["log_dist", "haversine", "dist_ratio", "route_dist", "weight_abs", "weight_clipped",
          "equip", "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]
FEATURES = STATIC + ["market_idx", "quote_signal"]
RPM_LOW, RPM_HIGH = 0.8, 6.0          # plausible $/mile band used to drop corrupted labels
T0 = pd.Timestamp("2025-01-01")

LGB_PARAMS = dict(objective="huber", alpha=0.15, n_estimators=700, learning_rate=0.04,
                  num_leaves=31, min_child_samples=40, subsample=0.8, subsample_freq=1,
                  colsample_bytree=0.8, verbose=-1, n_jobs=1)


def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 3958.8 * 2 * np.arcsin(np.sqrt(a))


def load(path):
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def route_distance_table(train: pd.DataFrame, min_n: int = 5) -> pd.Series:
    """Median listed distance per (pickup, delivery) route; only routes with >= min_n loads."""
    g = train.groupby(["pickup", "delivery"])["distance"].agg(["median", "count"])
    return g.loc[g["count"] >= min_n, "median"]


def daily_market_index(df: pd.DataFrame) -> pd.Series:
    """Mean market_index per calendar day (features only, no labels), gaps interpolated."""
    daily = df.groupby("date")["market_index"].mean()
    return daily.reindex(pd.date_range(daily.index.min(), daily.index.max())).interpolate()


def build_features(df: pd.DataFrame, route_tbl: pd.Series, use_route_dist: bool = True) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    X["log_dist"] = np.log(df["distance"])
    X["haversine"] = haversine(df.pickup_lat, df.pickup_lon, df.delivery_lat, df.delivery_lon)
    X["dist_ratio"] = df["distance"] / X["haversine"]
    keys = pd.MultiIndex.from_frame(df[["pickup", "delivery"]])
    rd = pd.Series(route_tbl.reindex(keys).values, index=df.index)
    X["route_dist"] = rd.fillna(df["distance"]) if use_route_dist else df["distance"]
    X["weight_abs"] = df["weight"].abs()                          # negative weights = sign errors
    X["weight_clipped"] = (X["weight_abs"] >= 47500).astype(int)  # values capped at 47,500
    X["equip"] = df["equipment"].map(EQUIP)
    for c in ["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]:
        X[c] = df[c]
    X["market_idx"] = df["market_index"].fillna(df["date"].map(daily_market_index(df)))
    X["quote_signal"] = df["quote_signal"]
    return X


def days_since(df):
    return (df["date"] - T0).dt.days.astype(float).values


def estimate_trend(train, X, y, ok):
    """Daily-median residual of a date-free model, regressed on market index + time -> per-day trend."""
    oof = np.zeros(len(train))
    p = {**LGB_PARAMS, "n_estimators": 300, "learning_rate": 0.06}
    for a, b in KFold(4, shuffle=True, random_state=0).split(X):
        m = lgb.LGBMRegressor(**p).fit(X.iloc[a][STATIC], y.iloc[a])
        oof[b] = m.predict(X.iloc[b][STATIC])
    d = pd.DataFrame({"res": (y - oof)[ok], "mi": X["market_idx"][ok],
                      "t": days_since(train)[ok.values], "date": train["date"][ok]})
    d = d.groupby("date").agg(res=("res", "median"), mi=("mi", "mean"), t=("t", "first")).dropna()
    A = np.c_[np.ones(len(d)), d.mi, d.t]
    coef, *_ = np.linalg.lstsq(A, d.res.values, rcond=None)
    return float(coef[2])


def clean_labels(train: pd.DataFrame) -> pd.Series:
    return (train["posted_rate"] / train["distance"]).between(RPM_LOW, RPM_HIGH)


class FreightModel:
    def __init__(self, use_route_dist=True, drop_bad_labels=True, use_trend=True, seeds=(0, 1, 2)):
        self.use_route_dist, self.drop_bad, self.use_trend, self.seeds = use_route_dist, drop_bad_labels, use_trend, seeds

    def fit(self, train: pd.DataFrame):
        self.route_tbl = route_distance_table(train)
        X = build_features(train, self.route_tbl, self.use_route_dist)
        y = np.log(train["posted_rate"])
        ok = clean_labels(train)
        keep = ok if self.drop_bad else pd.Series(True, index=train.index)
        self.gamma = estimate_trend(train, X, y, ok) if self.use_trend else 0.0
        off = self.gamma * days_since(train)
        self.models = [lgb.LGBMRegressor(**{**LGB_PARAMS, "random_state": s}).fit(X[keep][FEATURES], (y - off)[keep])
                       for s in self.seeds]
        return self

    def predict(self, df: pd.DataFrame, X: pd.DataFrame | None = None) -> np.ndarray:
        if X is None:
            X = build_features(df, self.route_tbl, self.use_route_dist)
        raw = np.mean([m.predict(X[FEATURES]) for m in self.models], axis=0)
        return np.exp(raw + self.gamma * days_since(df))


def metrics(pred, actual, distance):
    actual = np.asarray(actual, dtype=float)
    err = np.abs(pred - actual)
    clean = ((actual / np.asarray(distance)) >= RPM_LOW) & ((actual / np.asarray(distance)) <= RPM_HIGH)
    return {"MAE": err.mean(), "MAE_clean": err[clean].mean(), "MdAPE": np.median(err / actual),
            "MAPE": (err / actual).mean(), "RMSE": np.sqrt((err ** 2).mean())}
