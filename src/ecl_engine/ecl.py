from __future__ import annotations

import numpy as np
import pandas as pd


def add_performing_ecl(cube: pd.DataFrame) -> pd.DataFrame:
    frame = cube.copy()
    frame["calculation_method"] = np.where(frame["stage"].eq(1), "Stage 1 12-month default-event ECL", "Stage 2 lifetime default-event ECL")
    frame["period_ecl"] = frame["marginal_pd"] * frame["lgd"] * frame["ead"] * frame["discount_factor"]
    return frame


def build_stage3_cube(portfolio: pd.DataFrame, scenarios: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Stage 3 uses discounted expected cash shortfall, not the performing-loan PD engine."""
    loans = portfolio[portfolio["stage"].eq(3)].copy()
    if loans.empty:
        return pd.DataFrame()
    loans = loans.drop(columns=["unemployment_rate", "gdp_growth_yoy", "hpi_growth_yoy"], errors="ignore").rename(columns={"period": "reporting_period"})
    reporting_date = pd.Timestamp(cfg["reporting"]["reporting_date"])
    loans["months_in_default"] = (
        (reporting_date.to_period("M") - pd.to_datetime(loans["default_date"], errors="coerce").dt.to_period("M"))
        .apply(lambda x: x.n if pd.notna(x) else 0)
    ).clip(lower=0)
    assumed_total_resolution = int(cfg["lgd"]["recovery_months_incomplete_fallback"])
    loans["future_month"] = (assumed_total_resolution - loans["months_in_default"]).clip(lower=1).astype(int)
    scenario_names = scenarios[["scenario", "scenario_weight"]].drop_duplicates()
    rows = loans.assign(_key=1).merge(scenario_names.assign(_key=1), on="_key").drop(columns="_key")
    scen = scenarios.rename(columns={"month_number": "future_month", "period": "projection_period"})
    rows = rows.merge(scen.drop(columns="scenario_weight"), on=["scenario", "future_month"], how="left")
    rows["current_property_value"] = rows["current_actual_upb"] / (rows["current_ltv"] / 100).replace(0, np.nan)
    rows["current_property_value"] = rows["current_property_value"].fillna(rows["original_property_value"])
    rows["projected_property_value"] = rows["current_property_value"] * rows["hpi_factor"]
    rows["ead"] = rows["current_actual_upb"]
    rows["projected_ltv"] = (100 * rows["ead"] / rows["projected_property_value"].replace(0, np.nan)).clip(0, 500)
    rows["gross_collateral_proceeds"] = rows["projected_property_value"] * (1 - cfg["lgd"]["forced_sale_discount"])
    rows["recovery_expenses"] = rows["projected_property_value"] * cfg["lgd"]["foreclosure_cost_rate"]
    net_collateral = (rows["gross_collateral_proceeds"] - rows["recovery_expenses"]).clip(lower=0)
    mi_rate = rows["mortgage_insurance_percentage"].fillna(0).clip(0, 35) / 100
    rows["expected_mi_recovery"] = mi_rate * np.maximum(rows["ead"] - net_collateral, 0)
    rows["expected_recovery"] = np.minimum(rows["ead"], net_collateral + rows["expected_mi_recovery"])
    monthly_eir = rows["current_interest_rate"].fillna(0) / 100 / 12
    rows["discount_factor"] = 1 / ((1 + monthly_eir) ** rows["future_month"])
    rows["discounted_expected_recovery"] = rows["expected_recovery"] * rows["discount_factor"]
    rows["period_ecl"] = (rows["ead"] - rows["discounted_expected_recovery"]).clip(lower=0)
    rows["lgd"] = (rows["period_ecl"] / rows["ead"].replace(0, np.nan)).fillna(0).clip(0, 1)
    rows["conditional_pd"] = np.nan
    rows["survival_probability"] = np.nan
    rows["marginal_pd"] = np.nan
    rows["cumulative_pd"] = np.nan
    rows["raw_conditional_pd"] = np.nan
    rows["calculation_method"] = "Stage 3 discounted expected recovery cash shortfall"
    rows["ead_method"] = "current gross exposure at reporting date"
    rows["lgd_method"] = "scenario collateral realization, costs and MI recovery"
    rows["discount_rate_method"] = "loan-specific current contractual mortgage rate as EIR approximation"
    keep = ["loan_id", "stage", "scenario", "scenario_weight", "future_month", "period", "conditional_pd", "survival_probability", "marginal_pd", "cumulative_pd", "raw_conditional_pd", "ead", "projected_property_value", "projected_ltv", "gross_collateral_proceeds", "recovery_expenses", "expected_mi_recovery", "expected_recovery", "discounted_expected_recovery", "months_in_default", "lgd", "discount_factor", "period_ecl", "calculation_method", "ead_method", "lgd_method", "discount_rate_method", "unemployment_rate", "gdp_growth_yoy", "hpi_growth_yoy", "mortgage_rate"]
    rows = rows.rename(columns={"projection_period": "period"})
    return rows[keep]


def aggregate_ecl(cube: pd.DataFrame, portfolio: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scenario = cube.groupby(["loan_id", "stage", "scenario", "scenario_weight"], as_index=False)["period_ecl"].sum().rename(columns={"period_ecl": "scenario_ecl"})
    scenario["weighted_scenario_ecl"] = scenario["scenario_ecl"] * scenario["scenario_weight"]
    loan = scenario.groupby(["loan_id", "stage"], as_index=False)["weighted_scenario_ecl"].sum().rename(columns={"weighted_scenario_ecl": "loan_ecl"})
    loan = loan.merge(portfolio[[
        "loan_id", "current_actual_upb", "current_pd_12m", "origination_pd_12m", "current_lifetime_pd",
        "origination_lifetime_pd", "current_ltv", "current_dpd", "remaining_months_to_legal_maturity",
        "property_state", "origination_date", "primary_stage_reason",
    ]], on="loan_id", how="left")
    loan = loan.rename(columns={"current_actual_upb": "gross_exposure"})
    loan["coverage_ratio"] = loan["loan_ecl"] / loan["gross_exposure"].replace(0, np.nan)
    stage = loan.groupby("stage", as_index=False).agg(loans=("loan_id", "nunique"), gross_exposure=("gross_exposure", "sum"), ecl=("loan_ecl", "sum"))
    stage["coverage_ratio"] = stage["ecl"] / stage["gross_exposure"]
    return scenario, loan, stage


def worked_traces(cube: pd.DataFrame, loan_ecl: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = []
    for stage in (1, 2, 3):
        candidates = loan_ecl[loan_ecl["stage"].eq(stage)]
        if not candidates.empty:
            selected.append(candidates.sort_values("loan_ecl", ascending=False).iloc[0]["loan_id"])
    monthly = cube[cube["loan_id"].isin(selected)].copy()
    summary = loan_ecl[loan_ecl["loan_id"].isin(selected)].copy()
    summary["trace_explanation"] = summary["stage"].map({1: "Sum monthly MPD x LGD x EAD x DF through month 12 separately by scenario, then probability-weight scenario totals.", 2: "Sum monthly MPD x LGD x EAD x DF through remaining life separately by scenario, then probability-weight scenario totals.", 3: "Gross exposure less scenario-specific discounted expected collateral/MI recovery; no performing-loan PD curve."})
    return summary, monthly
