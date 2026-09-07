from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


LGD_NUMERIC = ["estimated_ltv_at_default", "ead_at_default", "original_property_value", "unemployment_rate", "hpi_growth_yoy", "mortgage_insurance_percentage", "loan_age_at_default"]
LGD_CATEGORICAL = ["property_state", "property_type", "occupancy_status", "modification_at_default"]


@dataclass
class LGDModel:
    pipeline: Pipeline | None
    fallback_lgd: float
    training_rows: int
    metrics: pd.DataFrame


def save_lgd_model(model: LGDModel, cfg: dict) -> None:
    path = Path(cfg["paths"]["models"])
    path.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path / "workout_lgd_model.joblib")


def prepare_lgd_development_data(recoveries: pd.DataFrame, macro: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """Join default, recovery, macroeconomic and loan-age fields used in LGD development."""
    data = recoveries.copy()
    data["modification_at_default"] = data["modification_at_default"].fillna("Unknown")
    data = data.merge(macro[["period", "unemployment_rate", "hpi_growth_yoy"]].rename(columns={"period": "default_date"}), on="default_date", how="left")
    ages = panel.loc[panel["default_event"].eq(1), ["loan_id", "loan_age"]].rename(columns={"loan_age": "loan_age_at_default"})
    data = data.merge(ages, on="loan_id", how="left")
    return data


def fit_lgd_model(recoveries: pd.DataFrame, macro: pd.DataFrame, panel: pd.DataFrame) -> LGDModel:
    """Fit a bounded logit-transformed ridge model on completed workout LGDs."""
    data = prepare_lgd_development_data(recoveries, macro, panel)
    completed = data[data["completed_workout"] & data["workout_lgd"].notna() & data["ead_at_default"].gt(0)].copy()
    fallback = float(completed["workout_lgd"].median()) if len(completed) else 0.35
    if len(completed) < 20 or completed["workout_lgd"].nunique() < 3:
        return LGDModel(None, fallback, len(completed), pd.DataFrame([{"sample": "completed_workouts", "rows": len(completed), "mae": np.nan, "method": "fallback median because model sample was insufficient"}]))
    completed = completed.sort_values(["default_date", "loan_id"]).reset_index(drop=True)
    dates = completed["default_date"].drop_duplicates().sort_values().reset_index(drop=True)
    cutoff = dates.iloc[min(max(int(len(dates) * 0.80), 1), len(dates) - 1)]
    development = completed[completed["default_date"] < cutoff].copy()
    out_of_time = completed[completed["default_date"] >= cutoff].copy()
    if len(development) < 20 or out_of_time.empty:
        split = min(max(20, int(len(completed) * 0.80)), len(completed) - 1)
        development, out_of_time = completed.iloc[:split].copy(), completed.iloc[split:].copy()
    y = np.log(development["workout_lgd"].clip(0.005, 0.995) / (1 - development["workout_lgd"].clip(0.005, 0.995)))
    prep = ColumnTransformer([
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), LGD_NUMERIC),
        ("categorical", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), LGD_CATEGORICAL),
    ])
    x = development[LGD_NUMERIC + LGD_CATEGORICAL].copy()
    x[LGD_NUMERIC] = x[LGD_NUMERIC].astype(float)
    for col in LGD_CATEGORICAL:
        x[col] = x[col].astype(object).where(x[col].notna(), np.nan)
    model = Pipeline([("prepare", prep), ("ridge", Ridge(alpha=5.0))]).fit(x, y)
    metrics = []
    for sample_name, sample in [("development", development), ("out_of_time", out_of_time)]:
        model_x = sample[LGD_NUMERIC + LGD_CATEGORICAL].copy()
        model_x[LGD_NUMERIC] = model_x[LGD_NUMERIC].astype(float)
        for col in LGD_CATEGORICAL:
            model_x[col] = model_x[col].astype(object).where(model_x[col].notna(), np.nan)
        prediction = 1 / (1 + np.exp(-model.predict(model_x)))
        actual = sample["workout_lgd"].to_numpy(float)
        benchmark = np.full(len(sample), development["workout_lgd"].median())
        metrics.append({
            "sample": sample_name,
            "rows": len(sample),
            "mae": mean_absolute_error(actual, prediction),
            "rmse": mean_squared_error(actual, prediction) ** 0.5,
            "r_squared": r2_score(actual, prediction) if len(sample) > 1 else np.nan,
            "observed_lgd": actual.mean(),
            "predicted_lgd": prediction.mean(),
            "median_benchmark_mae": mean_absolute_error(actual, benchmark),
            "method": "logit-transformed bounded ridge regression",
        })
    return LGDModel(model, fallback, len(development), pd.DataFrame(metrics))


def add_lgd_paths(cube: pd.DataFrame, portfolio: pd.DataFrame, lgd_model: LGDModel, cfg: dict) -> pd.DataFrame:
    frame = cube.copy()
    if frame.empty:
        for column in ["current_property_value", "projected_property_value", "projected_ltv", "lgd"]:
            frame[column] = pd.Series(dtype=float)
        frame["lgd_method"] = pd.Series(dtype=object)
        return frame
    current_property = portfolio[["loan_id", "current_actual_upb", "current_ltv", "property_state", "original_property_value", "mortgage_insurance_percentage"]].copy()
    current_property["current_property_value"] = current_property["current_actual_upb"] / (current_property["current_ltv"] / 100).replace(0, np.nan)
    current_property["current_property_value"] = current_property["current_property_value"].fillna(current_property["original_property_value"])
    frame = frame.merge(current_property.drop(columns=["current_actual_upb", "current_ltv"]), on="loan_id", how="left")
    frame["projected_property_value"] = frame["current_property_value"] * frame["hpi_factor"]
    frame["projected_ltv"] = (100 * frame["ead"] / frame["projected_property_value"].replace(0, np.nan)).clip(0, 500)
    base_drivers = frame.loc[frame["scenario"].eq("Base"), [
        "loan_id", "future_month", "projected_ltv", "unemployment_rate", "hpi_growth_yoy",
    ]].rename(columns={
        "projected_ltv": "base_projected_ltv", "unemployment_rate": "base_unemployment_rate",
        "hpi_growth_yoy": "base_hpi_growth_yoy",
    })
    frame = frame.merge(base_drivers, on=["loan_id", "future_month"], how="left")
    model_input = pd.DataFrame({
        "estimated_ltv_at_default": frame["base_projected_ltv"], "ead_at_default": frame["ead"],
        "original_property_value": frame["original_property_value"],
        "unemployment_rate": frame["base_unemployment_rate"], "hpi_growth_yoy": frame["base_hpi_growth_yoy"],
        "mortgage_insurance_percentage": frame["mortgage_insurance_percentage"], "loan_age_at_default": frame["loan_age"] + frame["future_month"],
        "property_state": frame["property_state"], "property_type": frame["property_type"], "occupancy_status": frame["occupancy_status"],
        "modification_at_default": np.where(frame["modification_history"].eq(1), "Y", "N"),
    })
    if lgd_model.pipeline is None:
        base = np.full(len(frame), lgd_model.fallback_lgd)
    else:
        x = model_input[LGD_NUMERIC + LGD_CATEGORICAL].copy()
        x[LGD_NUMERIC] = x[LGD_NUMERIC].astype(float)
        for col in LGD_CATEGORICAL:
            x[col] = x[col].astype(object).where(x[col].notna(), np.nan)
        raw = lgd_model.pipeline.predict(x)
        base = 1 / (1 + np.exp(-raw))
    collateral_floor = (1 - (frame["projected_property_value"] * (1 - cfg["lgd"]["forced_sale_discount"] - cfg["lgd"]["foreclosure_cost_rate"])) / frame["ead"].replace(0, np.nan)).fillna(0)
    frame["lgd"] = np.maximum(base, collateral_floor)
    frame["lgd"] = frame["lgd"].clip(cfg["lgd"]["floor"], cfg["lgd"]["cap"])
    frame["lgd_method"] = "completed-workout bounded regression on Base drivers with scenario collateral floor"
    return frame
