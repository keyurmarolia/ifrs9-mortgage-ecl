from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def estimate_behavioral_prepayment(panel: pd.DataFrame, cfg: dict) -> dict:
    cutoff = pd.Timestamp(cfg["pd"]["test_start_date"])
    development_panel = panel[panel["period"] < cutoff]
    exposure_years = max(development_panel["current_actual_upb"].gt(0).sum() / 12, 1)
    voluntary = development_panel["zero_balance_code"].fillna("").eq("01").sum()
    annual_rate = float(np.clip(voluntary / exposure_years, cfg["ead"]["annual_prepayment_floor"], cfg["ead"]["annual_prepayment_cap"]))
    smm = 1 - (1 - annual_rate) ** (1 / 12)
    # Compare observed balances immediately before default with contractual-like prior balances.
    ordered = panel.sort_values(["loan_id", "period"]).copy()
    ordered["prior_upb"] = ordered.groupby("loan_id")["current_actual_upb"].shift(1)
    default_rows = ordered[ordered["default_event"].eq(1) & ordered["period"].lt(cutoff)].copy()
    default_rows["prior_rate"] = ordered.groupby("loan_id")["current_interest_rate"].shift(1).loc[default_rows.index]
    default_rows["prior_remaining"] = ordered.groupby("loan_id")["remaining_months_to_legal_maturity"].shift(1).loc[default_rows.index]
    expected = contractual_balance(
        default_rows["prior_upb"].to_numpy(float), default_rows["prior_rate"].fillna(0).to_numpy(float),
        default_rows["prior_remaining"].fillna(1).to_numpy(float), np.ones(len(default_rows)),
    )
    ratio = (default_rows["current_actual_upb"].to_numpy(float) / np.where(expected > 0, expected, np.nan))
    ratio = pd.Series(ratio).replace([np.inf, -np.inf], np.nan)
    bad_factor = float(ratio.median()) if ratio.notna().any() else 1.0
    return {"annual_conditional_prepayment_rate": annual_rate, "monthly_single_mortality": smm, "balance_at_default_adjustment": float(np.clip(bad_factor, 0.95, 1.05))}


def _default_balance_sample(panel: pd.DataFrame, behavioral: dict, cfg: dict) -> pd.DataFrame:
    ordered = panel.sort_values(["loan_id", "period"]).copy()
    groups = ordered.groupby("loan_id", sort=False)
    ordered["prior_period"] = groups["period"].shift(1)
    ordered["prior_upb"] = groups["current_actual_upb"].shift(1)
    ordered["prior_rate"] = groups["current_interest_rate"].shift(1)
    ordered["prior_remaining"] = groups["remaining_months_to_legal_maturity"].shift(1)
    month_gap = (ordered["period"].dt.year - ordered["prior_period"].dt.year) * 12 + (ordered["period"].dt.month - ordered["prior_period"].dt.month)
    sample = ordered[ordered["default_event"].eq(1) & month_gap.eq(1) & ordered["prior_upb"].gt(0)].copy()
    sample["actual_default_balance"] = sample["current_actual_upb"].where(sample["current_actual_upb"].gt(0))
    sample["actual_default_balance"] = sample["actual_default_balance"].fillna(sample["zero_balance_removal_upb"].where(sample["zero_balance_removal_upb"].gt(0))).fillna(sample["prior_upb"])
    sample["predicted_default_balance"] = contractual_balance(
        sample["prior_upb"].to_numpy(float),
        sample["prior_rate"].fillna(0).to_numpy(float),
        sample["prior_remaining"].fillna(1).to_numpy(float),
        np.ones(len(sample)),
    ) * behavioral["balance_at_default_adjustment"]
    sample["split"] = np.where(sample["period"] < pd.Timestamp(cfg["pd"]["test_start_date"]), "development", "out_of_time")
    return sample


def ead_backtest(panel: pd.DataFrame, behavioral: dict, cfg: dict) -> pd.DataFrame:
    """Compare predicted and observed balances in the first default month."""
    sample = _default_balance_sample(panel, behavioral, cfg)
    rows = []
    for name, data in sample.groupby("split"):
        actual = data["actual_default_balance"].to_numpy(float)
        predicted = data["predicted_default_balance"].to_numpy(float)
        denominator = np.maximum(actual, 1000)
        rows.append({
            "sample": name,
            "rows": len(data),
            "mae_usd": mean_absolute_error(actual, predicted),
            "rmse_usd": mean_squared_error(actual, predicted) ** 0.5,
            "mape": float(np.mean(np.abs(actual - predicted) / denominator)),
            "weighted_absolute_percentage_error": float(np.abs(actual - predicted).sum() / max(actual.sum(), 1)),
            "mean_error_usd": float(np.mean(predicted - actual)),
            "target": "balance in first default month",
        })
    return pd.DataFrame(rows)


def ead_backtest_sample(panel: pd.DataFrame, behavioral: dict, cfg: dict, rows_per_split: int = 5000) -> pd.DataFrame:
    sample = _default_balance_sample(panel, behavioral, cfg)
    selected = [group.sample(min(len(group), rows_per_split), random_state=20260821) for _, group in sample.groupby("split")]
    return pd.concat(selected, ignore_index=True)[["loan_id", "period", "split", "prior_upb", "prior_rate", "prior_remaining", "actual_default_balance", "predicted_default_balance"]]


def ead_macro_sensitivity(features: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Test whether macro variables materially improve voluntary-prepayment ranking."""
    ordered = features.sort_values(["loan_id", "period"]).copy()
    ordered["next_voluntary_prepayment"] = ordered.groupby("loan_id")["zero_balance_code"].shift(-1).eq("01").fillna(False).astype(int)
    sample = ordered[ordered["survival_observation"] & ordered["next_month_observed"]].copy()
    cutoff = pd.Timestamp(cfg["pd"]["test_start_date"])
    train, test = sample[sample["period"] < cutoff], sample[sample["period"] >= cutoff]
    base_features = ["loan_age_sqrt", "remaining_months_to_legal_maturity", "current_actual_upb", "current_interest_rate", "current_ltv"]
    macro_features = base_features + ["unemployment_rate", "gdp_growth_yoy", "hpi_growth_yoy"]
    rows = []
    for name, variables in [("loan_only", base_features), ("loan_plus_macro", macro_features)]:
        prep = ColumnTransformer([("numeric", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), variables)])
        model = Pipeline([("prepare", prep), ("model", LogisticRegression(max_iter=300, class_weight="balanced", C=0.5))])
        model.fit(train[variables], train["next_voluntary_prepayment"])
        probability = model.predict_proba(test[variables])[:, 1]
        rows.append({"model": name, "sample": "out_of_time", "rows": len(test), "events": int(test["next_voluntary_prepayment"].sum()), "roc_auc": roc_auc_score(test["next_voluntary_prepayment"], probability)})
    result = pd.DataFrame(rows)
    base_auc = float(result.loc[result["model"].eq("loan_only"), "roc_auc"].iloc[0])
    macro_auc = float(result.loc[result["model"].eq("loan_plus_macro"), "roc_auc"].iloc[0])
    result["macro_auc_improvement"] = macro_auc - base_auc
    material = (macro_auc >= 0.60) and (macro_auc - base_auc >= 0.005)
    result["production_conclusion"] = "review scenario-sensitive EAD" if material else "retain scenario-invariant EAD"
    return result


def contractual_balance(principal: np.ndarray, annual_rate_pct: np.ndarray, remaining_months: np.ndarray, month: np.ndarray) -> np.ndarray:
    r = annual_rate_pct / 100 / 12
    n = np.maximum(remaining_months, 1)
    k = np.minimum(month, n)
    zero_rate = np.abs(r) < 1e-12
    payment = np.empty_like(principal, dtype=float)
    payment[zero_rate] = principal[zero_rate] / n[zero_rate]
    payment[~zero_rate] = principal[~zero_rate] * r[~zero_rate] / (1 - (1 + r[~zero_rate]) ** (-n[~zero_rate]))
    balance = np.empty_like(principal, dtype=float)
    balance[zero_rate] = principal[zero_rate] - payment[zero_rate] * k[zero_rate]
    balance[~zero_rate] = principal[~zero_rate] * (1 + r[~zero_rate]) ** k[~zero_rate] - payment[~zero_rate] * ((1 + r[~zero_rate]) ** k[~zero_rate] - 1) / r[~zero_rate]
    return np.maximum(balance, 0)


def build_projection_frame(portfolio: pd.DataFrame, scenarios: pd.DataFrame, behavioral: dict, cfg: dict) -> pd.DataFrame:
    """Generate Loan x Scenario x Month rows and fixed-rate contractual mortgage EAD."""
    max_months = int(cfg["reporting"]["maximum_projection_months"])
    performing = portfolio[portfolio["stage"].isin([1, 2])].copy()
    records = []
    scenario_names = scenarios["scenario"].drop_duplicates().tolist()
    for row in performing.itertuples(index=False):
        requested_horizon = int(max(row.ecl_horizon_months, 1))
        if requested_horizon > max_months:
            raise ValueError(f"Projection cap {max_months} is shorter than remaining life {requested_horizon} for {row.loan_id}")
        horizon = requested_horizon
        for scenario in scenario_names:
            records.append(pd.DataFrame({"loan_id": row.loan_id, "stage": row.stage, "scenario": scenario, "future_month": np.arange(1, horizon + 1, dtype=np.int16)}))
    cube = pd.concat(records, ignore_index=True) if records else pd.DataFrame(columns=["loan_id", "stage", "scenario", "future_month"])
    cube = cube.merge(scenarios.rename(columns={"month_number": "future_month"}), on=["scenario", "future_month"], how="left")
    loan_cols = ["loan_id", "current_actual_upb", "current_interest_rate", "remaining_months_to_legal_maturity", "current_ltv", "current_dpd", "max_dpd_3m", "max_dpd_6m", "max_dpd_12m", "loan_age", "loan_age_sqrt", "modification_history", "classic_fico", "original_dti", "original_ltv", "occupancy_status", "loan_purpose", "property_type", "current_non_interest_bearing_upb", "current_interest_bearing_upb"]
    cube = cube.merge(portfolio[loan_cols], on="loan_id", how="left")
    deferred = cube["current_non_interest_bearing_upb"].fillna(0).clip(lower=0)
    interest_bearing = cube["current_interest_bearing_upb"].fillna(cube["current_actual_upb"] - deferred).clip(lower=0)
    cube["interest_bearing_balance"] = contractual_balance(
        interest_bearing.to_numpy(float), cube["current_interest_rate"].fillna(0).to_numpy(float),
        cube["remaining_months_to_legal_maturity"].to_numpy(float), cube["future_month"].to_numpy(float),
    )
    cube["deferred_balance"] = np.where(cube["future_month"] < cube["remaining_months_to_legal_maturity"], deferred, 0.0)
    cube["contractual_balance"] = cube["interest_bearing_balance"] + cube["deferred_balance"]
    cube["ead"] = cube["contractual_balance"] * behavioral["balance_at_default_adjustment"]
    cube["ead_method"] = "conditional contractual balance including deferred UPB and observed balance-at-default adjustment; no CCF"
    return cube
