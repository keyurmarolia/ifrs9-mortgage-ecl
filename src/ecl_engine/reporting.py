from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import sqlite3
import numpy as np
import pandas as pd


def _risk_grade(pd12: pd.Series) -> pd.Categorical:
    return pd.cut(pd12, [-np.inf, .005, .015, .04, .10, np.inf], labels=["A: <=0.5%", "B: 0.5-1.5%", "C: 1.5-4%", "D: 4-10%", "E: >10%"])


def build_reporting_tables(portfolio, transitions, scenarios, cube, scenario_ecl, loan_ecl, stage_summary, pd_metrics, pd_woe_iv, pd_calibration, lgd_metrics, ead_backtest_metrics, ead_macro_metrics, scenario_weight_metrics, pd_macro_metrics, recoveries, controls, trace_summary, trace_monthly) -> dict[str, pd.DataFrame]:
    total_exposure = loan_ecl["gross_exposure"].sum()
    total_ecl = loan_ecl["loan_ecl"].sum()
    performing_loans = loan_ecl[loan_ecl["stage"].isin([1, 2])]
    performing_exposure = performing_loans["gross_exposure"].sum()
    performing_cube = cube[cube["stage"].isin([1, 2])].copy()
    performing_cube["lgd_driver_weight"] = (
        performing_cube["scenario_weight"] * performing_cube["marginal_pd"]
        * performing_cube["ead"] * performing_cube["discount_factor"]
    )
    lgd_denominator = performing_cube["lgd_driver_weight"].sum()
    ecl_driver_lgd = (
        (performing_cube["lgd"] * performing_cube["lgd_driver_weight"]).sum() / lgd_denominator
        if lgd_denominator else np.nan
    )
    executive = pd.DataFrame([
        {"metric": "Reporting date", "value": str(portfolio["period"].max().date()), "unit": "date"},
        {"metric": "Mortgage accounts", "value": len(loan_ecl), "unit": "count"},
        {"metric": "Gross exposure", "value": total_exposure, "unit": "USD"},
        {"metric": "Probability-weighted ECL", "value": total_ecl, "unit": "USD"},
        {"metric": "Portfolio coverage ratio", "value": total_ecl / total_exposure if total_exposure else np.nan, "unit": "percent"},
        {"metric": "Performing exposure-weighted 12-month PD", "value": np.average(performing_loans["current_pd_12m"], weights=performing_loans["gross_exposure"]) if performing_exposure else np.nan, "unit": "percent"},
    ])
    stage = stage_summary.copy()
    stage["exposure_share"] = stage["gross_exposure"] / total_exposure
    stage["ecl_share"] = stage["ecl"] / total_ecl
    scen = scenario_ecl.groupby(["scenario", "scenario_weight"], as_index=False)["scenario_ecl"].sum()
    scen["weighted_contribution"] = scen["scenario_ecl"] * scen["scenario_weight"]
    monthly = cube.groupby(["stage", "scenario", "future_month"], as_index=False).agg(
        ead=("ead", "sum"), marginal_pd=("marginal_pd", "mean"),
        lgd=("lgd", "mean"), period_ecl=("period_ecl", "sum"))
    monthly = monthly.rename(columns={"marginal_pd": "loan_average_marginal_pd", "lgd": "loan_average_lgd"})
    pd_dist = loan_ecl.assign(pd_band=pd.cut(loan_ecl["current_pd_12m"], [-np.inf, .005, .01, .02, .05, .10, np.inf], labels=["<=0.5%", "0.5-1%", "1-2%", "2-5%", "5-10%", ">10%"]))
    pd_dist = pd_dist.groupby("pd_band", observed=False, as_index=False).agg(loans=("loan_id", "count"), exposure=("gross_exposure", "sum"), ecl=("loan_ecl", "sum"))
    lgd_dist = cube.groupby("loan_id", as_index=False)["lgd"].mean().merge(loan_ecl[["loan_id", "gross_exposure", "loan_ecl"]], on="loan_id")
    lgd_dist["lgd_band"] = pd.cut(lgd_dist["lgd"], [0, .1, .2, .3, .4, .6, 1], include_lowest=True)
    lgd_dist = lgd_dist.groupby("lgd_band", observed=False, as_index=False).agg(loans=("loan_id", "count"), exposure=("gross_exposure", "sum"), ecl=("loan_ecl", "sum"))
    ltv = loan_ecl.assign(ltv_band=pd.cut(loan_ecl["current_ltv"], [0, 50, 60, 70, 80, 90, 100, 125, np.inf], include_lowest=True))
    ltv = ltv.groupby("ltv_band", observed=False, as_index=False).agg(loans=("loan_id", "count"), exposure=("gross_exposure", "sum"), ecl=("loan_ecl", "sum"))
    vintage = loan_ecl.assign(vintage=loan_ecl["origination_date"].dt.year).groupby("vintage", as_index=False).agg(loans=("loan_id", "count"), exposure=("gross_exposure", "sum"), ecl=("loan_ecl", "sum"))
    risk = performing_loans.assign(risk_grade=_risk_grade(performing_loans["current_pd_12m"])).groupby("risk_grade", observed=False, as_index=False).agg(loans=("loan_id", "count"), exposure=("gross_exposure", "sum"), ecl=("loan_ecl", "sum"))
    dpd = loan_ecl.assign(dpd_band=pd.cut(loan_ecl["current_dpd"].fillna(-1), [-np.inf, -0.1, 29, 59, 89, np.inf], labels=["Unknown", "Current/<30", "30-59", "60-89", "90+"])).groupby("dpd_band", observed=False, as_index=False).agg(loans=("loan_id", "count"), exposure=("gross_exposure", "sum"), ecl=("loan_ecl", "sum"))
    geography = loan_ecl.groupby("property_state", dropna=False, as_index=False).agg(loans=("loan_id", "count"), exposure=("gross_exposure", "sum"), ecl=("loan_ecl", "sum"))
    lifetime_dist = loan_ecl.assign(lifetime_pd_band=pd.cut(loan_ecl["current_lifetime_pd"], [-np.inf, .01, .05, .10, .25, .50, np.inf], labels=["<=1%", "1-5%", "5-10%", "10-25%", "25-50%", ">50%"]))
    lifetime_dist = lifetime_dist.groupby("lifetime_pd_band", observed=False, as_index=False).agg(loans=("loan_id", "count"), exposure=("gross_exposure", "sum"), ecl=("loan_ecl", "sum"))
    # EAD is scenario-invariant in this mortgage implementation; use one scenario to avoid triple counting.
    ead_profile = cube[cube["scenario"].eq("Base")].groupby(["stage", "future_month"], as_index=False)["ead"].sum()
    pd_term = performing_cube.groupby(["stage", "scenario", "future_month"], as_index=False).agg(
        conditional_pd=("conditional_pd", "mean"), survival_probability=("survival_probability", "mean"),
        marginal_pd=("marginal_pd", "mean"), cumulative_pd=("cumulative_pd", "mean"))
    lgd_term = cube.groupby(["stage", "scenario", "future_month"], as_index=False).agg(
        projected_property_value=("projected_property_value", "sum"), projected_ltv=("projected_ltv", "mean"), lgd=("lgd", "mean"))
    ead_term = cube.groupby(["stage", "scenario", "future_month"], as_index=False).agg(ead=("ead", "sum"))
    discount_term = cube.groupby(["stage", "scenario", "future_month"], as_index=False).agg(discount_factor=("discount_factor", "mean"))
    recovery_summary = recoveries.groupby("completed_workout", as_index=False).agg(defaults=("loan_id", "count"), ead_at_default=("ead_at_default", "sum"), discounted_recoveries=("total_discounted_recoveries", "sum"), average_lgd=("workout_lgd", "mean"))
    scenario_wide = scenario_ecl.pivot(index="loan_id", columns="scenario", values="scenario_ecl").add_suffix("_scenario_ecl").reset_index()
    account = loan_ecl.merge(scenario_wide, on="loan_id", how="left").sort_values(["stage", "loan_ecl"], ascending=[True, False]).copy()
    weighted_lgd = cube.copy()
    weighted_lgd["lgd_weight"] = np.where(
        weighted_lgd["stage"].isin([1, 2]),
        weighted_lgd["scenario_weight"] * weighted_lgd["marginal_pd"].fillna(0) * weighted_lgd["ead"] * weighted_lgd["discount_factor"],
        weighted_lgd["scenario_weight"] * weighted_lgd["ead"],
    )
    weighted_lgd["weighted_lgd"] = weighted_lgd["lgd"] * weighted_lgd["lgd_weight"]
    loan_lgd = weighted_lgd.groupby("loan_id", as_index=False).agg(weighted_lgd=("weighted_lgd", "sum"), lgd_weight=("lgd_weight", "sum"))
    loan_lgd["average_lgd"] = loan_lgd["weighted_lgd"] / loan_lgd["lgd_weight"].replace(0, np.nan)
    loan_lgd = loan_lgd[["loan_id", "average_lgd"]]
    account = account.merge(loan_lgd, on="loan_id", how="left")
    lgd_by_ltv = account.assign(ltv_band=pd.cut(account["current_ltv"], [0, 50, 60, 70, 80, 90, 100, 125, np.inf], include_lowest=True)).groupby("ltv_band", observed=False, as_index=False).agg(loans=("loan_id", "count"), exposure=("gross_exposure", "sum"), average_lgd=("average_lgd", "mean"), ecl=("loan_ecl", "sum"))
    executive = pd.concat([executive, pd.DataFrame([{"metric": "ECL-driver-weighted performing LGD", "value": ecl_driver_lgd, "unit": "percent"}])], ignore_index=True)
    # Public reporting tables use stable portfolio aliases. The local database retains source loan IDs
    # in its authoritative input and calculation tables.
    aliases = {loan_id: f"RM{number:06d}" for number, loan_id in enumerate(sorted(portfolio["loan_id"].astype(str).unique()), start=1)}
    account["loan_id"] = account["loan_id"].astype(str).map(aliases)
    trace_summary = trace_summary.copy()
    trace_monthly = trace_monthly.copy()
    trace_summary["loan_id"] = trace_summary["loan_id"].astype(str).map(aliases)
    trace_monthly["loan_id"] = trace_monthly["loan_id"].astype(str).map(aliases)
    return {
        "executive_summary": executive, "stage_summary": stage, "scenario_summary": scen,
        "monthly_ecl_term_structure": monthly, "transition_matrix": transitions,
        "pd_distribution": pd_dist, "lgd_distribution": lgd_dist, "ead_profile": ead_profile,
        "ltv_analysis": ltv, "lgd_by_ltv": lgd_by_ltv, "dpd_analysis": dpd, "geography_analysis": geography,
        "lifetime_pd_distribution": lifetime_dist, "vintage_analysis": vintage, "risk_grade_analysis": risk,
        "account_level_ecl": account, "macro_scenarios": scenarios, "pd_model_metrics": pd_metrics,
        "pd_woe_iv": pd_woe_iv, "pd_calibration": pd_calibration,
        "lgd_model_metrics": lgd_metrics, "ead_backtest_metrics": ead_backtest_metrics,
        "ead_macro_sensitivity": ead_macro_metrics, "scenario_weight_analysis": scenario_weight_metrics,
        "pd_macro_satellite_metrics": pd_macro_metrics,
        "pd_term_structure": pd_term, "lgd_term_structure": lgd_term,
        "ead_term_structure": ead_term, "discount_factor_term_structure": discount_term,
        "recovery_summary": recovery_summary,
        "controls": controls, "worked_trace_summary": trace_summary, "worked_trace_monthly": trace_monthly,
    }


def write_outputs(tables: dict[str, pd.DataFrame], database_tables: dict[str, pd.DataFrame], cfg: dict) -> None:
    table_dir = Path(cfg["paths"]["tables"])
    staging_dir = table_dir.with_name(table_dir.name + ".next")
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        frame.to_csv(staging_dir / f"{name}.csv", index=False)
    database = Path(cfg["paths"]["database"])
    database.parent.mkdir(parents=True, exist_ok=True)
    database_next = database.with_suffix(database.suffix + ".next")
    database_next.unlink(missing_ok=True)
    with sqlite3.connect(database_next) as connection:
        for name, frame in {**database_tables, **tables}.items():
            safe = frame.copy()
            if name == "ecl_projection_cube":
                preferred = [
                    "loan_id", "stage", "scenario", "scenario_weight", "future_month", "period",
                    "unemployment_rate", "gdp_growth_yoy", "hpi_growth_yoy", "mortgage_rate",
                    "raw_conditional_pd", "conditional_pd", "survival_probability", "marginal_pd", "cumulative_pd",
                    "ead", "projected_property_value", "projected_ltv", "lgd", "discount_factor", "period_ecl",
                ]
                safe = safe[[col for col in preferred if col in safe] + [col for col in safe.columns if col not in preferred]]
            for col in safe.columns:
                if pd.api.types.is_datetime64_any_dtype(safe[col]):
                    safe[col] = safe[col].dt.strftime("%Y-%m-%d")
                elif isinstance(safe[col].dtype, pd.PeriodDtype):
                    safe[col] = safe[col].astype("string")
                elif isinstance(safe[col].dtype, pd.CategoricalDtype):
                    safe[col] = safe[col].astype("string").astype(object)
                elif pd.api.types.is_extension_array_dtype(safe[col].dtype):
                    if pd.api.types.is_numeric_dtype(safe[col].dtype):
                        safe[col] = pd.to_numeric(safe[col], errors="coerce").astype(float)
                    else:
                        safe[col] = safe[col].astype(object)
            safe = safe.where(pd.notna(safe), None)
            safe.to_sql(name, connection, if_exists="replace", index=False, chunksize=50000)
        connection.execute("CREATE INDEX IF NOT EXISTS ix_cube_loan_scenario_month ON ecl_projection_cube (loan_id, scenario, future_month)")
        view_sql = {
            "ecl_projection_base": "SELECT * FROM ecl_projection_cube WHERE scenario='Base'",
            "ecl_projection_upside": "SELECT * FROM ecl_projection_cube WHERE scenario='Upside'",
            "ecl_projection_downside": "SELECT * FROM ecl_projection_cube WHERE scenario='Downside'",
            "pd_term_structure_detail": "SELECT loan_id,stage,scenario,future_month,conditional_pd,survival_probability,marginal_pd,cumulative_pd FROM ecl_projection_cube WHERE stage IN (1,2)",
            "lgd_term_structure_detail": "SELECT loan_id,stage,scenario,future_month,projected_property_value,projected_ltv,lgd FROM ecl_projection_cube",
            "ead_term_structure_detail": "SELECT loan_id,stage,scenario,future_month,ead FROM ecl_projection_cube",
            "discount_factor_term_structure_detail": "SELECT loan_id,stage,scenario,future_month,discount_factor FROM ecl_projection_cube",
        }
        for view_name, sql in view_sql.items():
            connection.execute(f"DROP VIEW IF EXISTS {view_name}")
            connection.execute(f"CREATE VIEW {view_name} AS {sql}")
    manifest = {name: {"rows": len(frame), "columns": list(frame.columns)} for name, frame in {**database_tables, **tables}.items()}
    (staging_dir / "output_manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    database_backup = database.with_suffix(database.suffix + ".previous")
    database_backup.unlink(missing_ok=True)
    if database.exists():
        os.replace(database, database_backup)
    os.replace(database_next, database)
    tables_backup = table_dir.with_name(table_dir.name + ".previous")
    if tables_backup.exists():
        shutil.rmtree(tables_backup)
    if table_dir.exists():
        os.replace(table_dir, tables_backup)
    os.replace(staging_dir, table_dir)
    database_backup.unlink(missing_ok=True)
    if tables_backup.exists():
        shutil.rmtree(tables_backup)
