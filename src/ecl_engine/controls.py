from __future__ import annotations

import numpy as np
import pandas as pd


def run_controls(origination, panel, macro, portfolio, scenarios, cube, loan_ecl, stage_summary) -> pd.DataFrame:
    def maximum_absolute(series: pd.Series) -> float:
        return float(series.abs().max()) if not series.empty else 0.0

    scenario_weights = scenarios.groupby("scenario")["scenario_weight"].first().sum()
    performing = cube[cube["stage"].isin([1, 2])].copy()
    finite_columns = ["ead", "lgd", "discount_factor", "period_ecl"]
    probability_columns = ["conditional_pd", "survival_probability", "marginal_pd", "cumulative_pd"]
    base12 = performing[(performing["scenario"].eq("Base")) & (performing["future_month"] <= 12)]
    base12_cpd = base12.groupby("loan_id")["marginal_pd"].sum()
    pd_target = portfolio.set_index("loan_id")["current_pd_12m"].reindex(base12_cpd.index)
    stage2_horizon = performing[performing["stage"].eq(2)].groupby("loan_id")["future_month"].max()
    expected_stage2 = portfolio.set_index("loan_id").loc[stage2_horizon.index, "ecl_horizon_months"]
    stage3 = cube[cube["stage"].eq(3)]
    controls = [
        ("C01", "Origination loan IDs unique", int(origination["loan_id"].duplicated().sum()), 0),
        ("C02", "Loan-month keys unique", int(panel.duplicated(["loan_id", "period"]).sum()), 0),
        ("C03", "Monthly chronology populated", int(panel["period"].isna().sum()), 0),
        ("C04", "Scenario weights sum to 100%", round(float(scenario_weights), 12), 1.0),
        ("C05", "Every reporting loan has one valid stage", int((~portfolio["stage"].isin([1, 2, 3])).sum()), 0),
        ("C06", "Projection probabilities within bounds", int(((cube["conditional_pd"].dropna() < 0) | (cube["conditional_pd"].dropna() > 1)).sum()), 0),
        ("C07", "LGD within bounds", int(((cube["lgd"] < 0) | (cube["lgd"] > 1)).sum()), 0),
        ("C08", "EAD non-negative", int((cube["ead"] < -1e-8).sum()), 0),
        ("C09", "Discount factors in (0,1]", int(((cube["discount_factor"] <= 0) | (cube["discount_factor"] > 1.0000001)).sum()), 0),
        ("C10", "Portfolio ECL equals sum of loan ECL", round(float(stage_summary["ecl"].sum() - loan_ecl["loan_ecl"].sum()), 6), 0.0),
        ("C11", "All source rows explicitly non-synthetic", int(origination["synthetic_field_flag"].sum()), 0),
        ("C12", "Macro history available", int(macro.empty), 0),
        ("C13", "Projection keys unique", int(cube.duplicated(["loan_id", "scenario", "future_month"]).sum()), 0),
        ("C14", "All loans have every configured scenario", int((cube.groupby("loan_id")["scenario"].nunique() != scenarios["scenario"].nunique()).sum()), 0),
        ("C15", "Core projection values finite", int((~np.isfinite(cube[finite_columns].to_numpy(float))).sum()), 0),
        ("C16", "Performing probability values finite", int((~np.isfinite(performing[probability_columns].to_numpy(float))).sum()), 0),
        ("C17", "Marginal PD equals survival times conditional PD", maximum_absolute(performing["marginal_pd"] - performing["survival_probability"] * performing["conditional_pd"]), 0.0),
        ("C18", "Base 12-month cumulative PD reconciles to scored PD", maximum_absolute(base12_cpd - pd_target), 0.0),
        ("C19", "Stage 2 horizon equals remaining modelled lifetime", int((stage2_horizon.to_numpy() != expected_stage2.to_numpy()).sum()), 0),
        ("C20", "Performing period ECL formula reconciles", maximum_absolute(performing["period_ecl"] - performing["marginal_pd"] * performing["lgd"] * performing["ead"] * performing["discount_factor"]), 0.0),
        ("C21", "Stage 3 cash shortfall formula reconciles", float((stage3["period_ecl"] - (stage3["ead"] - stage3["discounted_expected_recovery"]).clip(lower=0)).abs().max()) if not stage3.empty else 0.0, 0.0),
        ("C22", "Scenario weights are non-negative", int((scenarios.groupby("scenario")["scenario_weight"].first() < 0).sum()), 0),
        ("C23", "Reporting snapshot is exact configured date", int((portfolio["period"] != pd.Timestamp(scenarios["period"].min()) - pd.offsets.MonthBegin(1)).sum()), 0),
    ]
    if not performing.empty:
        scenario_order = {"Upside": 0, "Base": 1, "Downside": 2}
        ordered = performing[performing["scenario"].isin(scenario_order)].copy()
        ordered["scenario_order"] = ordered["scenario"].map(scenario_order)
        monotonic = ordered.pivot_table(index=["loan_id", "future_month"], columns="scenario", values="conditional_pd")
        violations = 0
        if {"Upside", "Base", "Downside"}.issubset(monotonic.columns):
            violations = int(((monotonic["Upside"] > monotonic["Base"] + 1e-12) | (monotonic["Base"] > monotonic["Downside"] + 1e-12)).sum())
        controls.append(("C24", "Conditional PD ordering is Upside <= Base <= Downside", violations, 0))
    scenario_totals = cube.groupby("scenario")["period_ecl"].sum()
    ecl_ordering_violations = 0
    if {"Upside", "Base", "Downside"}.issubset(scenario_totals.index):
        ecl_ordering_violations = int(
            (scenario_totals["Upside"] > scenario_totals["Base"] + 1e-6)
            or (scenario_totals["Base"] > scenario_totals["Downside"] + 1e-6)
        )
    controls.append(("C25", "Scenario ECL ordering is Upside <= Base <= Downside", ecl_ordering_violations, 0))
    result = pd.DataFrame(controls, columns=["control_id", "description", "actual", "expected"])
    result["status"] = np.where(np.isclose(result["actual"].astype(float), result["expected"].astype(float), atol=1e-5), "PASS", "FAIL")
    return result
