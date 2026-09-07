from __future__ import annotations

import numpy as np
import pandas as pd

from .ead import contractual_balance
from .features import CATEGORICAL_PD_FEATURES
from .lifetime_pd import _calibration_exponents
from .pd_model import HAZARD_NUMERIC_FEATURES
from .pd_model import score_12m


def _origination_feature_frame(portfolio: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    base = portfolio.copy()
    base["current_ltv"] = base["original_ltv"]
    base["current_dpd"] = 0
    base[["max_dpd_3m", "max_dpd_6m", "max_dpd_12m"]] = 0
    base["loan_age"] = 0
    base["loan_age_sqrt"] = 0.0
    base["remaining_months_to_legal_maturity"] = base["original_loan_term"]
    base["current_actual_upb"] = base["original_upb"]
    base["current_interest_rate"] = base["original_interest_rate"]
    base["modification_history"] = 0
    macro_lookup = macro.sort_values("period").set_index("period")
    for idx, row in base.iterrows():
        eligible = macro_lookup.loc[:row["origination_date"]]
        selected = eligible.iloc[-1] if not eligible.empty else macro_lookup.iloc[0]
        for col in ("unemployment_rate", "gdp_growth_yoy", "hpi_growth_yoy"):
            base.at[idx, col] = selected[col]
    return base


def _sicr_lifetime_pd(
    portfolio: pd.DataFrame, hazard_model, macro_satellite, scenarios: pd.DataFrame,
    behavioral: dict, cfg: dict, at_origination: bool,
) -> pd.Series:
    """Calculate a hazard-based lifetime PD over the same remaining horizon for SICR."""
    records = []
    for row in portfolio.itertuples(index=False):
        horizon = max(int(row.remaining_months_to_legal_maturity), 1)
        records.append(pd.DataFrame({"loan_id": row.loan_id, "future_month": np.arange(1, horizon + 1)}))
    frame = pd.concat(records, ignore_index=True)
    loan_columns = [
        "loan_id", "classic_fico", "original_dti", "original_ltv", "original_upb",
        "original_interest_rate", "original_loan_term", "original_property_value",
        "occupancy_status", "loan_purpose", "property_type", "current_actual_upb",
        "current_interest_rate", "remaining_months_to_legal_maturity", "current_ltv",
        "current_dpd", "max_dpd_3m", "max_dpd_6m", "max_dpd_12m", "loan_age",
        "modification_history", "current_pd_12m", "origination_pd_12m",
    ]
    frame = frame.merge(portfolio[loan_columns], on="loan_id", how="left")
    base_macro = scenarios[scenarios["scenario"].eq("Base")].rename(columns={"month_number": "future_month"})
    frame = frame.merge(base_macro[["future_month", "unemployment_rate", "gdp_growth_yoy", "hpi_growth_yoy", "hpi_factor"]], on="future_month", how="left")
    if at_origination:
        principal = frame["original_upb"].to_numpy(float)
        rate = frame["original_interest_rate"].fillna(0).to_numpy(float)
        maturity = frame["original_loan_term"].fillna(1).to_numpy(float)
        frame["current_ltv"] = 100 * contractual_balance(principal, rate, maturity, frame["future_month"].to_numpy(float)) / frame["original_property_value"].replace(0, np.nan)
        frame["current_actual_upb"] = contractual_balance(principal, rate, maturity, frame["future_month"].to_numpy(float))
        frame["loan_age"] = frame["future_month"]
        frame["remaining_months_to_legal_maturity"] = (frame["original_loan_term"] - frame["future_month"]).clip(lower=0)
        frame[["current_dpd", "max_dpd_3m", "max_dpd_6m", "max_dpd_12m", "modification_history"]] = 0.0
    else:
        principal = frame["current_actual_upb"].to_numpy(float)
        rate = frame["current_interest_rate"].fillna(0).to_numpy(float)
        maturity = frame["remaining_months_to_legal_maturity"].fillna(1).to_numpy(float)
        projected_balance = contractual_balance(principal, rate, maturity, frame["future_month"].to_numpy(float))
        current_property = frame["current_actual_upb"] / (frame["current_ltv"] / 100).replace(0, np.nan)
        projected_property = current_property.fillna(frame["original_property_value"]) * frame["hpi_factor"]
        frame["current_actual_upb"] = projected_balance
        frame["current_ltv"] = 100 * projected_balance / projected_property.replace(0, np.nan)
        frame["loan_age"] = frame["loan_age"] + frame["future_month"]
        frame["remaining_months_to_legal_maturity"] = (frame["remaining_months_to_legal_maturity"] - frame["future_month"]).clip(lower=0)
        decay = np.exp(-frame["future_month"] / 6.0)
        frame["current_dpd"] = frame["current_dpd"].fillna(0) * decay
        frame["max_dpd_3m"] = np.maximum(frame["current_dpd"], frame["max_dpd_3m"].fillna(0) * np.exp(-frame["future_month"] / 3.0))
        frame["max_dpd_6m"] = np.maximum(frame["max_dpd_3m"], frame["max_dpd_6m"].fillna(0) * np.exp(-frame["future_month"] / 6.0))
        frame["max_dpd_12m"] = np.maximum(frame["max_dpd_6m"], frame["max_dpd_12m"].fillna(0) * np.exp(-frame["future_month"] / 12.0))
    frame["loan_age_sqrt"] = np.sqrt(frame["loan_age"].clip(lower=0))
    raw = hazard_model.predict_proba(frame[HAZARD_NUMERIC_FEATURES + CATEGORICAL_PD_FEATURES])[:, 1]
    raw = np.clip(raw, cfg["pd"]["probability_floor"], cfg["pd"]["probability_cap"])
    if at_origination:
        frame["raw_conditional_pd"] = raw
        target = portfolio.set_index("loan_id")["origination_pd_12m"]
    else:
        adjustment = macro_satellite.adjustment(frame)
        odds = raw / (1 - raw) * np.exp(np.clip(adjustment, -5, 5))
        frame["raw_conditional_pd"] = np.clip(odds / (1 + odds), cfg["pd"]["probability_floor"], cfg["pd"]["probability_cap"])
        target = portfolio.set_index("loan_id")["current_pd_12m"]
    base12 = frame[frame["future_month"] <= 12]
    exponent = _calibration_exponents(base12, target, float(behavioral["monthly_single_mortality"]))
    frame = frame.merge(exponent, left_on="loan_id", right_index=True, how="left")
    frame["conditional_pd"] = 1 - (1 - frame["raw_conditional_pd"]) ** frame["hazard_calibration_exponent"]
    monthly_survival = (1 - frame["conditional_pd"]) * (1 - float(behavioral["monthly_single_mortality"]))
    frame["survival"] = monthly_survival.groupby(frame["loan_id"]).transform(lambda x: x.cumprod().shift(1, fill_value=1.0))
    frame["marginal_pd"] = frame["survival"] * frame["conditional_pd"]
    return frame.groupby("loan_id")["marginal_pd"].sum().clip(upper=1)


def assign_stages(portfolio: pd.DataFrame, pd12_model, hazard_model, macro_satellite, scenarios: pd.DataFrame, behavioral: dict, macro: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Compare hazard-based current and origination lifetime risk and apply backstops."""
    result = portfolio.copy()
    result["current_pd_12m"] = score_12m(pd12_model, result, cfg)
    origin_features = _origination_feature_frame(result, macro)
    result["origination_pd_12m"] = score_12m(pd12_model, origin_features, cfg)
    result["current_lifetime_pd"] = result["loan_id"].map(_sicr_lifetime_pd(result, hazard_model, macro_satellite, scenarios, behavioral, cfg, False))
    result["origination_lifetime_pd"] = result["loan_id"].map(_sicr_lifetime_pd(result, hazard_model, macro_satellite, scenarios, behavioral, cfg, True))
    result["lifetime_pd_ratio"] = result["current_lifetime_pd"] / result["origination_lifetime_pd"].replace(0, np.nan)
    result["lifetime_pd_absolute_change"] = result["current_lifetime_pd"] - result["origination_lifetime_pd"]
    result["trigger_default"] = result["default_flag"].eq(1)
    result["trigger_30dpd_backstop"] = result["current_dpd"] >= int(cfg["sicr"]["dpd_backstop"])
    result["trigger_modification"] = result["modification_history"].eq(1) & bool(cfg["sicr"]["modification_is_qualitative_trigger"])
    result["trigger_prior_default"] = result["prior_default_history"].fillna(False) & bool(cfg["sicr"]["prior_default_is_qualitative_trigger"])
    result["trigger_relative_pd"] = result["lifetime_pd_ratio"] >= float(cfg["sicr"]["relative_pd_multiple"])
    result["trigger_absolute_pd"] = result["lifetime_pd_absolute_change"] >= float(cfg["sicr"]["absolute_pd_increase"])
    sicr = result[["trigger_30dpd_backstop", "trigger_modification", "trigger_prior_default", "trigger_relative_pd", "trigger_absolute_pd"]].any(axis=1)
    result["stage"] = np.select([result["trigger_default"], sicr], [3, 2], default=1).astype(int)
    result["primary_stage_reason"] = np.select(
        [result["trigger_default"], result["trigger_30dpd_backstop"], result["trigger_prior_default"], result["trigger_modification"], result["trigger_absolute_pd"], result["trigger_relative_pd"]],
        ["Current default/credit-impaired", "30 DPD backstop", "Prior default history", "Modification qualitative indicator", "Absolute lifetime-PD deterioration", "Relative lifetime-PD deterioration"],
        default="No SICR trigger")
    result["ecl_horizon_months"] = np.where(result["stage"].eq(1), np.minimum(12, result["remaining_months_to_legal_maturity"]), result["remaining_months_to_legal_maturity"]).clip(min=1)
    result["pd_reporting_status"] = np.where(result["stage"].eq(3), "Not applicable: already defaulted", "Performing exposure PD")
    result.loc[result["stage"].eq(3), [
        "current_pd_12m", "current_lifetime_pd", "lifetime_pd_ratio",
        "lifetime_pd_absolute_change",
    ]] = np.nan
    return result
