from __future__ import annotations

import numpy as np
import pandas as pd
import re
from sklearn.metrics import roc_auc_score


def _plain_interval_labels(series: pd.Series) -> pd.Series:
    def label(value) -> str:
        text = str(value)
        return re.sub(
            r"-?\d+(?:\.\d+)?e[+-]?\d+",
            lambda match: f"{float(match.group()):.10f}".rstrip("0").rstrip("."),
            text,
            flags=re.IGNORECASE,
        )
    return series.astype(object).map(label).astype("string")


def ks_statistic(y_true, probability) -> float:
    frame = pd.DataFrame({"target": y_true, "probability": probability}).dropna()
    if frame["target"].nunique() < 2:
        return np.nan
    grouped = frame.groupby("probability", as_index=False)["target"].agg(["sum", "count"]).reset_index().sort_values("probability", ascending=False)
    defaults = grouped["sum"].sum()
    non_defaults = grouped["count"].sum() - defaults
    cumulative_defaults = grouped["sum"].cumsum() / defaults
    cumulative_non_defaults = (grouped["count"] - grouped["sum"]).cumsum() / non_defaults
    return float((cumulative_defaults - cumulative_non_defaults).abs().max())


def model_performance(y_true, probability) -> dict:
    y = pd.Series(y_true).astype(int)
    p = np.asarray(probability, dtype=float)
    auc = roc_auc_score(y, p) if y.nunique() > 1 else np.nan
    return {
        "roc_auc": float(auc) if np.isfinite(auc) else np.nan,
        "gini": float(2 * auc - 1) if np.isfinite(auc) else np.nan,
        "ks": ks_statistic(y, p),
    }


def woe_iv_table(frame: pd.DataFrame, target: str, variables: list[str], bins: int = 10) -> pd.DataFrame:
    """Calculate training-sample WOE and IV for binary default targets."""
    rows = []
    total_good = max(int((frame[target] == 0).sum()), 1)
    total_bad = max(int((frame[target] == 1).sum()), 1)
    for variable in variables:
        series = frame[variable]
        if pd.api.types.is_numeric_dtype(series) and series.nunique(dropna=True) > bins:
            try:
                cut = pd.qcut(series, q=bins, duplicates="drop")
                grouped = _plain_interval_labels(cut).where(cut.notna(), "Missing")
            except ValueError:
                grouped = series.astype(str)
        else:
            grouped = series.astype("string")
        grouped = grouped.fillna("Missing").replace("<NA>", "Missing")
        counts = pd.DataFrame({"bin": grouped, "target": frame[target]}).groupby("bin", dropna=False)["target"].agg(["count", "sum"]).reset_index()
        counts = counts.rename(columns={"sum": "bad"})
        counts["good"] = counts["count"] - counts["bad"]
        counts["distribution_good"] = (counts["good"] + 0.5) / (total_good + 0.5 * len(counts))
        counts["distribution_bad"] = (counts["bad"] + 0.5) / (total_bad + 0.5 * len(counts))
        counts["woe"] = np.log(counts["distribution_good"] / counts["distribution_bad"])
        counts["iv_component"] = (counts["distribution_good"] - counts["distribution_bad"]) * counts["woe"]
        counts.insert(0, "variable", variable)
        counts["default_rate"] = counts["bad"] / counts["count"].replace(0, np.nan)
        counts["variable_iv"] = counts["iv_component"].sum()
        rows.append(counts)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def calibration_table(y_true, probability, groups: int = 10) -> pd.DataFrame:
    frame = pd.DataFrame({"target": y_true, "predicted_pd": probability}).dropna()
    try:
        frame["band"] = pd.qcut(frame["predicted_pd"], groups, duplicates="drop")
    except ValueError:
        frame["band"] = "All"
    result = frame.groupby("band", observed=False, as_index=False).agg(
        observations=("target", "size"), defaults=("target", "sum"),
        observed_rate=("target", "mean"), predicted_rate=("predicted_pd", "mean")
    )
    result["calibration_ratio"] = result["predicted_rate"] / result["observed_rate"].replace(0, np.nan)
    def probability_band(value):
        if isinstance(value, pd.Interval):
            lower = max(float(value.left), 0.0)
            upper = max(float(value.right), 0.0)
            return f"{lower:.4%} to {upper:.4%}"
        return str(value)
    result["band"] = result["band"].map(probability_band)
    return result
