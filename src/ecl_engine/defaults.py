from __future__ import annotations

import numpy as np
import pandas as pd


def delinquency_to_dpd(value: object) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).strip().upper()
    if text in {"", "NAN", "XX", "<NA>"}:
        return np.nan
    if text == "RA":
        return 90.0
    try:
        months = int(float(text))
        return float(max(0, months * 30))
    except ValueError:
        return np.nan


def add_default_definition(performance: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Default is first 90+ DPD, REO, or a configured credit-related terminal event."""
    frame = performance.copy()
    frame["current_dpd"] = frame["current_delinquency_status"].map(delinquency_to_dpd)
    terminal = frame["zero_balance_code"].fillna("").isin(set(cfg["default"]["terminal_credit_codes"]))
    reo = frame["current_delinquency_status"].fillna("").str.upper().eq("RA")
    frame["default_flag"] = ((frame["current_dpd"].fillna(-1) >= int(cfg["default"]["dpd_threshold"])) | terminal | reo).astype("int8")
    first_default = frame.loc[frame["default_flag"].eq(1)].groupby("loan_id")["period"].min().rename("default_date")
    frame = frame.merge(first_default, on="loan_id", how="left")
    frame["default_event"] = (frame["period"].eq(frame["default_date"]) & frame["default_date"].notna()).astype("int8")
    state = pd.cut(frame["current_dpd"], [-1, 29, 59, 89, np.inf], labels=["Current/<30", "30-59", "60-89", "Default/90+"])
    frame["delinquency_state"] = state.astype("string").fillna("Unknown")
    frame.loc[frame["default_flag"].eq(1), "delinquency_state"] = "Default/90+"
    frame["delinquency_state"] = pd.Categorical(
        frame["delinquency_state"],
        categories=["Current/<30", "30-59", "60-89", "Default/90+", "Unknown"],
    )
    return frame


def build_recovery_table(panel: pd.DataFrame, origination: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    defaults = panel.loc[panel["default_event"].eq(1), ["loan_id", "period", "current_actual_upb", "current_dpd", "estimated_ltv", "modification_flag"]].copy()
    defaults.columns = ["loan_id", "default_date", "ead_at_default", "dpd_at_default", "estimated_ltv_at_default", "modification_at_default"]
    history = panel[panel["loan_id"].isin(defaults["loan_id"])].sort_values(["loan_id", "period"]).copy()
    terminal_rows = history[history["zero_balance_code"].notna()].groupby("loan_id", as_index=False).tail(1)
    terminal_rows = terminal_rows[[
        "loan_id", "net_sales_proceeds", "mi_recoveries", "non_mi_recoveries", "total_expenses",
        "zero_balance_effective_date", "zero_balance_code", "zero_balance_removal_upb", "actual_loss",
    ]].rename(columns={
        "net_sales_proceeds": "net_sale_proceeds", "non_mi_recoveries": "other_recoveries",
        "total_expenses": "recovery_expenses", "zero_balance_effective_date": "disposition_date",
    })
    workout = terminal_rows
    out = defaults.merge(workout, on="loan_id", how="left").merge(
        origination[["loan_id", "property_state", "property_type", "occupancy_status", "original_property_value", "mortgage_insurance_percentage", "original_interest_rate"]],
        on="loan_id", how="left")
    cash_cols = ["net_sale_proceeds", "mi_recoveries", "other_recoveries", "recovery_expenses"]
    out[cash_cols] = out[cash_cols].fillna(0.0)
    out["estimated_ltv_at_default"] = pd.to_numeric(out["estimated_ltv_at_default"], errors="coerce").replace(999, np.nan)
    out["recovery_duration_months"] = ((out["disposition_date"].dt.to_period("M") - out["default_date"].dt.to_period("M")).apply(lambda x: x.n if pd.notna(x) else np.nan))
    credit_resolution_codes = {"01", "02", "03", "09", "15"}
    out["completed_workout"] = out["disposition_date"].notna() & out["zero_balance_code"].isin(credit_resolution_codes)
    out["realized_loss_observed"] = out["actual_loss"].notna()
    duration = out["recovery_duration_months"].fillna(cfg["lgd"]["recovery_months_incomplete_fallback"]).clip(lower=1)
    monthly_rate = (out["original_interest_rate"].fillna(out["original_interest_rate"].median()) / 100) / 12
    reported_cash_recovery = (
        out["net_sale_proceeds"] + out["mi_recoveries"] + out["other_recoveries"] + out["recovery_expenses"]
    ).clip(lower=0)
    loss_implied_recovery = (out["ead_at_default"] - out["actual_loss"]).clip(lower=0)
    paid_in_full = out["zero_balance_code"].eq("01") & out["completed_workout"]
    observed_recovery = np.where(
        out["realized_loss_observed"], loss_implied_recovery,
        np.where(paid_in_full, out["ead_at_default"], reported_cash_recovery),
    )
    out["discounted_net_recoveries_observed"] = observed_recovery / ((1 + monthly_rate) ** duration)
    out["recovery_cashflow_source"] = np.select(
        [out["realized_loss_observed"], paid_in_full, out["completed_workout"]],
        ["Freddie actual loss converted to implied recovery", "Full payoff after historical default", "Reported disposition cash flows"],
        default="Modelled ultimate recovery for unresolved default")
    completed_recovery_rate = (out.loc[out["completed_workout"], "discounted_net_recoveries_observed"] / out.loc[out["completed_workout"], "ead_at_default"].replace(0, np.nan)).clip(0, 1)
    fallback = float(completed_recovery_rate.median()) if completed_recovery_rate.notna().any() else 0.65
    completed_duration = out.loc[out["completed_workout"] & out["recovery_duration_months"].notna(), "recovery_duration_months"]
    expected_resolution = float(completed_duration.median()) if completed_duration.notna().any() else float(cfg["lgd"]["recovery_months_incomplete_fallback"])
    panel_end = panel["period"].max()
    out["workout_age_months"] = (
        (panel_end.to_period("M") - out["default_date"].dt.to_period("M")).apply(lambda x: x.n if pd.notna(x) else np.nan)
    ).clip(lower=0)
    out["estimated_remaining_recoveries"] = 0.0
    incomplete = ~out["completed_workout"]
    collateral_proxy = (out["ead_at_default"] / (out["estimated_ltv_at_default"] / 100).replace(0, np.nan)).fillna(out["original_property_value"])
    undiscounted_estimate = np.minimum(
        out["ead_at_default"] * fallback,
        collateral_proxy * (1 - cfg["lgd"]["forced_sale_discount"] - cfg["lgd"]["foreclosure_cost_rate"]),
    ).clip(lower=0)
    remaining_months = (expected_resolution - out["workout_age_months"]).clip(lower=1)
    discounted_estimate = undiscounted_estimate / ((1 + monthly_rate) ** remaining_months)
    out.loc[incomplete, "estimated_remaining_recoveries"] = discounted_estimate[incomplete]
    out["total_discounted_recoveries"] = out["discounted_net_recoveries_observed"] + out["estimated_remaining_recoveries"]
    out["workout_lgd"] = (1 - out["total_discounted_recoveries"] / out["ead_at_default"].replace(0, np.nan)).clip(0, 1)
    out["expected_resolution_months"] = expected_resolution
    out["synthetic_data_flag"] = False
    out["model_estimate_flag"] = incomplete
    return out
