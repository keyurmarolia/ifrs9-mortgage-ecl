from __future__ import annotations

import numpy as np
import pandas as pd

from .features import CATEGORICAL_PD_FEATURES
from .pd_model import HAZARD_NUMERIC_FEATURES


def _calibration_exponents(base12: pd.DataFrame, targets: pd.Series, monthly_prepayment: float) -> pd.Series:
    """Solve the competing-risk 12-month default probability for each loan by bisection."""
    values = {}
    for loan_id, group in base12.groupby("loan_id", sort=False):
        raw = group.sort_values("future_month")["raw_conditional_pd"].to_numpy(float)
        target = float(targets.get(loan_id, np.nan))
        if not np.isfinite(target):
            values[loan_id] = 1.0
            continue

        def cumulative_default(exponent: float) -> float:
            q = 1 - (1 - raw) ** exponent
            survival = 1.0
            total = 0.0
            for hazard in q:
                total += survival * hazard
                survival *= (1 - hazard) * (1 - monthly_prepayment)
            return total

        lower, upper = 1e-10, 1.0
        while cumulative_default(upper) < target and upper < 1e8:
            upper *= 2
        for _ in range(60):
            midpoint = (lower + upper) / 2
            if cumulative_default(midpoint) < target:
                lower = midpoint
            else:
                upper = midpoint
        values[loan_id] = (lower + upper) / 2
    return pd.Series(values, name="hazard_calibration_exponent")


def add_lifetime_pd_paths(cube: pd.DataFrame, portfolio: pd.DataFrame, hazard_model, macro_satellite, behavioral: dict, cfg: dict) -> pd.DataFrame:
    """Predict monthly conditional hazard and transform it to survival, marginal PD and cumulative PD."""
    frame = cube.copy()
    if frame.empty:
        for column in ["raw_conditional_pd", "macro_log_odds_adjustment", "hazard_calibration_exponent", "conditional_pd", "survival_probability", "marginal_pd", "cumulative_pd", "prepayment_probability"]:
            frame[column] = pd.Series(dtype=float)
        return frame
    # Keep the loan-level hazard path on the Base collateral trajectory. Scenario variation
    # then enters once through the separately estimated macroeconomic satellite model.
    base_ltv = frame.loc[frame["scenario"].eq("Base"), ["loan_id", "future_month", "projected_ltv"]].rename(
        columns={"projected_ltv": "hazard_projected_ltv"}
    )
    frame = frame.merge(base_ltv, on=["loan_id", "future_month"], how="left")
    frame["current_ltv"] = frame["hazard_projected_ltv"].fillna(frame["projected_ltv"])
    frame["current_actual_upb"] = frame["ead"]
    frame["loan_age"] = frame["loan_age"] + frame["future_month"]
    frame["loan_age_sqrt"] = np.sqrt(frame["loan_age"].clip(lower=0))
    frame["remaining_months_to_legal_maturity"] = (frame["remaining_months_to_legal_maturity"] - frame["future_month"]).clip(lower=0)
    decay = np.exp(-frame["future_month"] / 6.0)
    frame["current_dpd"] = frame["current_dpd"].fillna(0) * decay
    frame["max_dpd_3m"] = np.maximum(frame["current_dpd"], frame["max_dpd_3m"].fillna(0) * np.exp(-frame["future_month"] / 3.0))
    frame["max_dpd_6m"] = np.maximum(frame["max_dpd_3m"], frame["max_dpd_6m"].fillna(0) * np.exp(-frame["future_month"] / 6.0))
    frame["max_dpd_12m"] = np.maximum(frame["max_dpd_6m"], frame["max_dpd_12m"].fillna(0) * np.exp(-frame["future_month"] / 12.0))
    raw = hazard_model.predict_proba(frame[HAZARD_NUMERIC_FEATURES + CATEGORICAL_PD_FEATURES])[:, 1]
    raw = np.clip(raw, cfg["pd"]["probability_floor"], cfg["pd"]["probability_cap"])
    frame["macro_log_odds_adjustment"] = macro_satellite.adjustment(frame)
    raw_odds = raw / (1 - raw)
    scenario_raw = raw_odds * np.exp(frame["macro_log_odds_adjustment"].clip(-5, 5))
    frame["raw_conditional_pd"] = np.clip(scenario_raw / (1 + scenario_raw), cfg["pd"]["probability_floor"], cfg["pd"]["probability_cap"])
    # Calibrate Base hazards exactly to the loan-level 12-month PD; apply the same exponent to every scenario.
    current_pd = portfolio.set_index("loan_id")["current_pd_12m"]
    base12 = frame[(frame["scenario"].eq("Base")) & (frame["future_month"] <= 12)].copy()
    exponent = _calibration_exponents(base12, current_pd, float(behavioral["monthly_single_mortality"]))
    frame = frame.merge(exponent, left_on="loan_id", right_index=True, how="left")
    frame["conditional_pd"] = 1 - (1 - frame["raw_conditional_pd"]) ** frame["hazard_calibration_exponent"]
    frame["conditional_pd"] = frame["conditional_pd"].clip(cfg["pd"]["probability_floor"], 1 - 1e-12)
    frame = frame.sort_values(["loan_id", "scenario", "future_month"])
    frame["prepayment_probability"] = float(behavioral["monthly_single_mortality"])
    monthly_survival = (1 - frame["conditional_pd"]) * (1 - frame["prepayment_probability"])
    frame["survival_probability"] = monthly_survival.groupby([frame["loan_id"], frame["scenario"]]).transform(
        lambda x: x.cumprod().shift(1, fill_value=1.0)
    )
    frame["marginal_pd"] = frame["survival_probability"] * frame["conditional_pd"]
    frame["cumulative_pd"] = frame.groupby(["loan_id", "scenario"])["marginal_pd"].cumsum().clip(upper=1)
    return frame
