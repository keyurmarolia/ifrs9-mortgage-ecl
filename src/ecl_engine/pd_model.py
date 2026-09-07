from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import brier_score_loss, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .features import NUMERIC_PD_FEATURES, CATEGORICAL_PD_FEATURES
from .model_diagnostics import calibration_table, model_performance, woe_iv_table

PD12_NUMERIC_FEATURES = [
    "classic_fico", "original_dti", "original_ltv", "current_ltv", "current_dpd",
    "max_dpd_3m", "max_dpd_6m", "max_dpd_12m", "current_actual_upb",
    "current_interest_rate", "modification_history", "unemployment_rate", "gdp_growth_yoy", "hpi_growth_yoy",
]

HAZARD_NUMERIC_FEATURES = [x for x in NUMERIC_PD_FEATURES if x not in {"unemployment_rate", "gdp_growth_yoy", "hpi_growth_yoy"}]


@dataclass
class MacroSatellite:
    model: Pipeline
    reference_adjustment: float
    metrics: pd.DataFrame

    @staticmethod
    def design(frame: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({
            "unemployment_rate": frame["unemployment_rate"].astype(float),
            "weak_gdp_growth": -frame["gdp_growth_yoy"].astype(float),
            "weak_hpi_growth": -frame["hpi_growth_yoy"].astype(float),
        }, index=frame.index)

    def adjustment(self, frame: pd.DataFrame) -> np.ndarray:
        return self.model.predict(self.design(frame)) - self.reference_adjustment


class OddsCalibratedModel:
    """Apply a transparent prevalence/odds calibration to a logistic score."""
    def __init__(self, base: Pipeline, odds_multiplier: float):
        self.base = base
        self.odds_multiplier = float(odds_multiplier)

    @property
    def named_steps(self):
        return self.base.named_steps

    def predict_proba(self, x):
        raw = self.base.predict_proba(x)[:, 1]
        odds = raw / np.clip(1 - raw, 1e-12, None)
        calibrated = (odds * self.odds_multiplier) / (1 + odds * self.odds_multiplier)
        return np.column_stack([1 - calibrated, calibrated])


def _pipeline(class_weight="balanced", numeric_features=None, categorical_features=None) -> Pipeline:
    numeric_features = numeric_features or NUMERIC_PD_FEATURES
    categorical_features = categorical_features or CATEGORICAL_PD_FEATURES
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    prep = ColumnTransformer([("numeric", numeric, numeric_features), ("categorical", categorical, categorical_features)])
    return Pipeline([("prepare", prep), ("model", LogisticRegression(max_iter=500, class_weight=class_weight, C=0.5))])


@dataclass
class PDModels:
    pd12: Pipeline
    hazard: Pipeline
    metrics: pd.DataFrame
    woe_iv: pd.DataFrame
    calibration: pd.DataFrame
    macro_satellite: MacroSatellite
    macro_metrics: pd.DataFrame


def _fit_macro_satellite(sample: pd.DataFrame, hazard_model: Pipeline, cfg: dict) -> MacroSatellite:
    """Fit a non-negative macro calibration to monthly hazard residuals."""
    loan_features = HAZARD_NUMERIC_FEATURES + CATEGORICAL_PD_FEATURES
    scored = sample.copy()
    scored["baseline_hazard"] = hazard_model.predict_proba(scored[loan_features])[:, 1]
    monthly = scored.groupby("period", as_index=False).agg(
        observations=("next_default_event", "size"), defaults=("next_default_event", "sum"),
        baseline_hazard=("baseline_hazard", "mean"), unemployment_rate=("unemployment_rate", "first"),
        gdp_growth_yoy=("gdp_growth_yoy", "first"), hpi_growth_yoy=("hpi_growth_yoy", "first"),
    )
    monthly["observed_hazard"] = (monthly["defaults"] + 0.5) / (monthly["observations"] + 1.0)
    observed_logit = np.log(monthly["observed_hazard"] / (1 - monthly["observed_hazard"]))
    baseline = monthly["baseline_hazard"].clip(1e-8, 1 - 1e-8)
    monthly["log_odds_residual"] = observed_logit - np.log(baseline / (1 - baseline))
    design = MacroSatellite.design(monthly)
    calibration_start = pd.Timestamp(cfg["pd"]["calibration_start_date"])
    test_start = pd.Timestamp(cfg["pd"]["test_start_date"])
    development = monthly[monthly["period"] < calibration_start]
    test = monthly[monthly["period"] >= test_start]
    prep = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    model = Pipeline([("prepare", prep), ("ridge", Ridge(alpha=10.0, positive=True))])
    model.fit(design.loc[development.index], development["log_odds_residual"], ridge__sample_weight=development["observations"])
    reference = float(model.predict(design.loc[development.index]).mean())
    rows = []
    for name, data in [("development", development), ("out_of_time", test)]:
        prediction = model.predict(design.loc[data.index])
        actual = data["log_odds_residual"].to_numpy(float)
        rows.append({
            "sample": name, "months": len(data), "defaults": int(data["defaults"].sum()),
            "mae_log_odds": mean_absolute_error(actual, prediction) if len(data) else np.nan,
            "r_squared": r2_score(actual, prediction) if len(data) > 1 else np.nan,
        })
    coefficients = model.named_steps["ridge"].coef_
    for variable, coefficient in zip(design.columns, coefficients):
        rows.append({"sample": "coefficient", "variable": variable, "coefficient": float(coefficient)})
    return MacroSatellite(model, reference, pd.DataFrame(rows))


def fit_pd_models(features: pd.DataFrame, cfg: dict) -> PDModels:
    """Fit interpretable logistic models for 12-month default and monthly conditional hazard."""
    sample = features[features["survival_observation"] & features["next_month_observed"]].copy()
    frequency = int(cfg["pd"]["training_row_frequency_months"])
    if frequency > 1:
        sample = sample[sample["period"].dt.month.sub(1).mod(frequency).eq(0)].copy()
    latest_complete_12m = min(features["period"].max(), pd.Timestamp(cfg["reporting"]["reporting_date"])) - pd.DateOffset(months=12)
    pd_sample = sample[sample["period"] <= latest_complete_12m].copy()
    calibration_start = pd.Timestamp(cfg["pd"]["calibration_start_date"])
    pd_cutoff = pd.Timestamp(cfg["pd"]["test_start_date"])
    pd_train = pd_sample[pd_sample["period"] < calibration_start]
    pd_calibration = pd_sample[(pd_sample["period"] >= calibration_start) & (pd_sample["period"] < pd_cutoff)]
    pd_test = pd_sample[pd_sample["period"] >= pd_cutoff]
    hazard_cutoff = pd.Timestamp(cfg["pd"]["test_start_date"])
    hazard_train = sample[sample["period"] < hazard_cutoff]
    hazard_test = sample[sample["period"] >= hazard_cutoff]
    if pd_train["target_default_12m"].nunique() < 2 or hazard_train["next_default_event"].nunique() < 2:
        raise ValueError("Insufficient default variation to fit PD models")
    pd12_features = PD12_NUMERIC_FEATURES + CATEGORICAL_PD_FEATURES
    pd12_base = _pipeline(cfg["pd"]["class_weight"], PD12_NUMERIC_FEATURES, CATEGORICAL_PD_FEATURES).fit(pd_train[pd12_features], pd_train["target_default_12m"])
    calibration_raw = pd12_base.predict_proba(pd_calibration[pd12_features])[:, 1]
    observed = float(pd_calibration["target_default_12m"].mean())
    predicted = float(calibration_raw.mean())
    odds_multiplier = (observed / max(1 - observed, 1e-12)) / (predicted / max(1 - predicted, 1e-12))
    pd12 = OddsCalibratedModel(pd12_base, odds_multiplier)
    hazard_features = HAZARD_NUMERIC_FEATURES + CATEGORICAL_PD_FEATURES
    hazard = _pipeline(cfg["pd"]["class_weight"], HAZARD_NUMERIC_FEATURES, CATEGORICAL_PD_FEATURES).fit(hazard_train[hazard_features], hazard_train["next_default_event"])
    macro_satellite = _fit_macro_satellite(sample[sample["period"] <= pd.Timestamp(cfg["reporting"]["reporting_date"])], hazard, cfg)
    metrics = []
    model_sets = [
        ("12-month logistic PD", pd12, "target_default_12m", pd_train, pd_test),
        ("discrete-time monthly hazard", hazard, "next_default_event", hazard_train, hazard_test),
    ]
    for name, model, target, train, test in model_sets:
        for split_name, data in [("development", train), ("out_of_time", test)]:
            variables = pd12_features if name.startswith("12-month") else hazard_features
            pred = model.predict_proba(data[variables])[:, 1]
            performance = model_performance(data[target], pred)
            metrics.append({"model": name, "sample": split_name, "rows": len(data), "defaults": int(data[target].sum()), **performance, "brier_score": brier_score_loss(data[target], pred), "observed_rate": data[target].mean(), "predicted_rate": pred.mean()})
    screening_variables = list(dict.fromkeys(PD12_NUMERIC_FEATURES + CATEGORICAL_PD_FEATURES))
    pd_woe = woe_iv_table(pd_train, "target_default_12m", screening_variables).assign(model="12-month logistic PD")
    hazard_woe = woe_iv_table(hazard_train, "next_default_event", hazard_features).assign(model="discrete-time monthly hazard")
    pd_calibration = calibration_table(pd_test["target_default_12m"], pd12.predict_proba(pd_test[pd12_features])[:, 1]).assign(model="12-month logistic PD", sample="out_of_time")
    hazard_calibration = calibration_table(hazard_test["next_default_event"], hazard.predict_proba(hazard_test[hazard_features])[:, 1]).assign(model="discrete-time monthly hazard", sample="out_of_time")
    return PDModels(
        pd12=pd12,
        hazard=hazard,
        metrics=pd.DataFrame(metrics),
        woe_iv=pd.concat([pd_woe, hazard_woe], ignore_index=True),
        calibration=pd.concat([pd_calibration, hazard_calibration], ignore_index=True),
        macro_satellite=macro_satellite,
        macro_metrics=macro_satellite.metrics,
    )


def build_notebook_development_sample(features: pd.DataFrame, cfg: dict, negative_ratio: int = 10) -> pd.DataFrame:
    """Create a compact weighted extract that reproduces the time splits used in PD development notebooks."""
    sample = features[features["survival_observation"] & features["next_month_observed"]].copy()
    latest_complete = min(features["period"].max(), pd.Timestamp(cfg["reporting"]["reporting_date"])) - pd.DateOffset(months=12)
    calibration_start = pd.Timestamp(cfg["pd"]["calibration_start_date"])
    test_start = pd.Timestamp(cfg["pd"]["test_start_date"])
    random_seed = int(cfg["project"]["random_seed"])
    parts = []
    specifications = [
        ("12_month_pd", "target_default_12m", sample[sample["period"] <= latest_complete], PD12_NUMERIC_FEATURES + CATEGORICAL_PD_FEATURES),
        ("monthly_hazard", "next_default_event", sample, HAZARD_NUMERIC_FEATURES + CATEGORICAL_PD_FEATURES),
    ]
    for target_name, target, population, variables in specifications:
        population = population.copy()
        population["split"] = np.select(
            [population["period"] < calibration_start, population["period"] < test_start],
            ["development", "calibration"], default="out_of_time",
        )
        for split, group in population.groupby("split"):
            positive = group[group[target].eq(1)]
            negative = group[group[target].eq(0)]
            n_negative = min(len(negative), max(len(positive) * negative_ratio, 2_000))
            chosen_negative = negative.sample(n=n_negative, random_state=random_seed) if n_negative < len(negative) else negative
            chosen = pd.concat([positive, chosen_negative]).copy()
            chosen["observation_weight"] = 1.0
            if len(chosen_negative):
                chosen.loc[chosen[target].eq(0), "observation_weight"] = len(negative) / len(chosen_negative)
            chosen["target_name"] = target_name
            chosen["target"] = chosen[target].astype(int)
            parts.append(chosen[["loan_id", "period", "target_name", "split", "target", "observation_weight", *variables]])
    return pd.concat(parts, ignore_index=True)


def score_12m(model: Pipeline, frame: pd.DataFrame, cfg: dict) -> np.ndarray:
    pred = model.predict_proba(frame[NUMERIC_PD_FEATURES + CATEGORICAL_PD_FEATURES])[:, 1]
    return np.clip(pred, cfg["pd"]["probability_floor"], cfg["pd"]["maximum_12m_pd"])


def save_models(models: PDModels, cfg: dict) -> None:
    path = Path(cfg["paths"]["models"])
    path.mkdir(parents=True, exist_ok=True)
    joblib.dump(models.pd12, path / "pd12_logistic.joblib")
    joblib.dump(models.hazard, path / "monthly_hazard_logistic.joblib")
    joblib.dump(models.macro_satellite, path / "pd_macro_satellite.joblib")


def coefficient_table(model: Pipeline, model_name: str) -> pd.DataFrame:
    names = model.named_steps["prepare"].get_feature_names_out()
    values = model.named_steps["model"].coef_[0]
    return pd.DataFrame({"model": model_name, "feature": names, "coefficient": values}).sort_values("coefficient")
