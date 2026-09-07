from __future__ import annotations

import numpy as np
import pandas as pd

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


def assign_stages(portfolio: pd.DataFrame, pd12_model, macro: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Compare current forward-looking lifetime risk with retained origination risk and apply backstops."""
    result = portfolio.copy()
    result["current_pd_12m"] = score_12m(pd12_model, result, cfg)
    origin_features = _origination_feature_frame(result, macro)
    result["origination_pd_12m"] = score_12m(pd12_model, origin_features, cfg)
    years = result["remaining_months_to_legal_maturity"].fillna(0).clip(lower=1) / 12
    result["current_lifetime_pd"] = 1 - (1 - result["current_pd_12m"]) ** years
    result["origination_lifetime_pd"] = 1 - (1 - result["origination_pd_12m"]) ** years
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
    return result
