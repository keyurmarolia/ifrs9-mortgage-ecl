from __future__ import annotations

import numpy as np
import pandas as pd


NUMERIC_PD_FEATURES = [
    "classic_fico", "original_dti", "original_ltv", "current_ltv", "current_dpd",
    "max_dpd_3m", "max_dpd_6m", "max_dpd_12m", "loan_age_sqrt",
    "remaining_months_to_legal_maturity", "current_actual_upb", "current_interest_rate",
    "modification_history", "unemployment_rate", "gdp_growth_yoy", "hpi_growth_yoy",
]

CATEGORICAL_PD_FEATURES = ["occupancy_status", "loan_purpose", "property_type"]


def build_monthly_features(panel: pd.DataFrame, origination: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    """Create borrower, loan, behaviour, collateral and macro features at each observation month."""
    origin_cols = ["loan_id", "origination_date", "classic_fico", "original_dti", "original_ltv", "original_upb",
                   "original_interest_rate", "original_loan_term", "occupancy_status", "loan_purpose",
                   "property_type", "property_state", "original_property_value", "mortgage_insurance_percentage"]
    frame = panel.merge(origination[origin_cols], on="loan_id", how="left").sort_values(["loan_id", "period"]).reset_index(drop=True)
    groups = frame.groupby("loan_id", sort=False)
    for months in (3, 6, 12):
        frame[f"max_dpd_{months}m"] = groups["current_dpd"].transform(lambda x: x.shift(1).rolling(months, min_periods=1).max()).fillna(0)
    frame["modification_history"] = groups["modification_flag"].transform(lambda x: x.fillna("N").isin(["Y", "P"]).cummax()).astype(int)
    frame["loan_age_sqrt"] = np.sqrt(frame["loan_age"].fillna(0).clip(lower=0))
    frame["balance_to_original_upb"] = frame["current_actual_upb"] / frame["original_upb"].replace(0, np.nan)
    observed_ltv = pd.to_numeric(frame["estimated_ltv"], errors="coerce").astype(float).replace(999, np.nan)
    amortized_ltv = (pd.to_numeric(frame["original_ltv"], errors="coerce").astype(float) * frame["balance_to_original_upb"].astype(float))
    frame["current_ltv"] = observed_ltv.fillna(amortized_ltv).clip(0, 300)
    frame = frame.merge(macro[["period", "unemployment_rate", "gdp_growth_yoy", "hpi_growth_yoy"]], on="period", how="left")
    if frame[["unemployment_rate", "gdp_growth_yoy", "hpi_growth_yoy"]].isna().any().any():
        missing = frame.loc[frame[["unemployment_rate", "gdp_growth_yoy", "hpi_growth_yoy"]].isna().any(axis=1), "period"].drop_duplicates()
        raise ValueError(f"Macro history is unavailable for {len(missing)} model months")
    groups = frame.groupby("loan_id", sort=False)
    frame["next_period"] = groups["period"].shift(-1)
    next_event = groups["default_event"].shift(-1)
    month_gap = (frame["next_period"].dt.year - frame["period"].dt.year) * 12 + (frame["next_period"].dt.month - frame["period"].dt.month)
    frame["next_month_observed"] = month_gap.eq(1)
    frame["next_default_event"] = np.where(frame["next_month_observed"], next_event, np.nan)
    frame["target_default_12m"] = 0
    has_default = frame["default_date"].notna()
    months_to_default = (frame.loc[has_default, "default_date"].dt.year - frame.loc[has_default, "period"].dt.year) * 12 + (frame.loc[has_default, "default_date"].dt.month - frame.loc[has_default, "period"].dt.month)
    frame.loc[has_default, "target_default_12m"] = months_to_default.between(1, 12).astype(int)
    frame["survival_observation"] = frame["default_date"].isna() | (frame["period"] < frame["default_date"])
    frame["prior_default_history"] = frame["default_date"].notna() & (frame["period"] > frame["default_date"])
    return frame


def reporting_portfolio(features: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    reporting_date = pd.Timestamp(cfg["reporting"]["reporting_date"])
    eligible = features[features["period"] <= reporting_date].sort_values(["loan_id", "period"]).groupby("loan_id", as_index=False).tail(1)
    eligible = eligible[eligible["period"].eq(reporting_date)].copy()
    eligible = eligible[(eligible["current_actual_upb"] > 0) & eligible["zero_balance_code"].isna()].copy()
    max_loans = int(cfg["data"]["reporting_portfolio_max_loans"])
    if len(eligible) > max_loans:
        high_risk = eligible[(eligible["current_dpd"] >= 30) | eligible["modification_history"].eq(1)]
        remaining = eligible.drop(high_risk.index)
        n = max(0, max_loans - len(high_risk))
        eligible = pd.concat([high_risk, remaining.sample(n=min(n, len(remaining)), random_state=int(cfg["project"]["random_seed"]))])
    return eligible.sort_values("loan_id").reset_index(drop=True)
