#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pandas as pd

from src.ecl_engine.config import load_config, ensure_directories
from src.ecl_engine.ingestion import load_origination, select_model_loan_ids, load_performance, load_macro_history
from src.ecl_engine.defaults import add_default_definition, build_recovery_table
from src.ecl_engine.features import build_monthly_features, reporting_portfolio
from src.ecl_engine.pd_model import fit_pd_models, save_models, coefficient_table, build_notebook_development_sample
from src.ecl_engine.transitions import transition_matrix
from src.ecl_engine.scenarios import build_scenarios, calibrate_scenario_weights
from src.ecl_engine.staging import assign_stages
from src.ecl_engine.ead import estimate_behavioral_prepayment, build_projection_frame, ead_backtest, ead_macro_sensitivity, ead_backtest_sample
from src.ecl_engine.lgd import fit_lgd_model, add_lgd_paths, save_lgd_model, prepare_lgd_development_data
from src.ecl_engine.discounting import add_discount_factors
from src.ecl_engine.lifetime_pd import add_lifetime_pd_paths
from src.ecl_engine.ecl import add_performing_ecl, build_stage3_cube, aggregate_ecl, worked_traces
from src.ecl_engine.controls import run_controls
from src.ecl_engine.reporting import build_reporting_tables, write_outputs
from src.ecl_engine.excel_reporting import build_excel_report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the independent end-to-end IFRS 9 mortgage ECL engine")
    parser.add_argument("--skip-workbook", action="store_true")
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    cfg = load_config(args.config)
    ensure_directories(cfg)

    print("1/12 Ingesting Freddie Mac origination and performance data", flush=True)
    origination = load_origination(cfg)
    model_ids = select_model_loan_ids(origination, cfg)
    performance = load_performance(cfg, model_ids)
    macro = load_macro_history(cfg)

    print("2/12 Applying default definition and constructing recovery workouts", flush=True)
    panel = add_default_definition(performance, cfg)
    recoveries = build_recovery_table(panel, origination, cfg)

    print("3/12 Engineering loan-month features", flush=True)
    features = build_monthly_features(panel, origination, macro)
    transitions = transition_matrix(panel)

    print("4/12 Fitting 12-month PD and discrete-time hazard models", flush=True)
    pd_models = fit_pd_models(features, cfg)
    pd_development_sample = build_notebook_development_sample(features, cfg)
    pd_coefficients = pd.concat([coefficient_table(pd_models.pd12, "12-month PD"), coefficient_table(pd_models.hazard, "monthly hazard")])

    print("5/12 Selecting the reporting-date active portfolio", flush=True)
    portfolio = reporting_portfolio(features, cfg)

    print("6/12 Building Base/Upside/Downside macroeconomic scenarios", flush=True)
    scenarios = build_scenarios(macro, portfolio, cfg)
    scenarios, scenario_weight_metrics = calibrate_scenario_weights(macro, scenarios, cfg)

    behavioral = estimate_behavioral_prepayment(panel, cfg)
    print("7/12 Calculating hazard-based SICR measures and assigning IFRS 9 stages", flush=True)
    portfolio = assign_stages(
        portfolio, pd_models.pd12, pd_models.hazard, pd_models.macro_satellite,
        scenarios, behavioral, macro, cfg,
    )

    print("8/12 Fitting workout LGD and validating EAD behaviour", flush=True)
    lgd_model = fit_lgd_model(recoveries, macro, panel)
    lgd_development_sample = prepare_lgd_development_data(recoveries, macro, panel)
    ead_backtest_metrics = ead_backtest(panel, behavioral, cfg)
    ead_development_sample = ead_backtest_sample(panel, behavioral, cfg)
    ead_macro_metrics = ead_macro_sensitivity(features, cfg)

    print("9/12 Building Loan x Scenario x Month EAD and LGD paths", flush=True)
    cube = build_projection_frame(portfolio, scenarios, behavioral, cfg)
    cube = add_lgd_paths(cube, portfolio, lgd_model, cfg)
    cube = add_discount_factors(cube)

    print("10/12 Producing monthly conditional, survival, marginal and cumulative PD paths", flush=True)
    cube = add_lifetime_pd_paths(cube, portfolio, pd_models.hazard, pd_models.macro_satellite, behavioral, cfg)
    cube = add_performing_ecl(cube)
    stage3 = build_stage3_cube(portfolio, scenarios, cfg)
    if not stage3.empty:
        cube = pd.concat([cube, stage3], ignore_index=True, sort=False)

    print("11/12 Aggregating scenario, loan, stage and portfolio ECL", flush=True)
    scenario_ecl, loan_ecl, stage_summary = aggregate_ecl(cube, portfolio)
    trace_summary, trace_monthly = worked_traces(cube, loan_ecl)
    controls = run_controls(origination, panel, macro, portfolio, scenarios, cube, loan_ecl, stage_summary, recoveries)
    if controls["status"].eq("FAIL").any():
        raise RuntimeError("Controls failed:\n" + controls.loc[controls["status"].eq("FAIL")].to_string(index=False))
    print("12/12 Writing SQLite, reporting tables and workbook", flush=True)
    tables = build_reporting_tables(
        portfolio, transitions, scenarios, cube, scenario_ecl, loan_ecl, stage_summary,
        pd_models.metrics, pd_models.woe_iv, pd_models.calibration, lgd_model.metrics,
        ead_backtest_metrics, ead_macro_metrics, scenario_weight_metrics, pd_models.macro_metrics,
        recoveries, controls, trace_summary, trace_monthly,
    )
    database_tables = {
        "origination_mortgages": origination, "monthly_loan_performance": panel,
        "default_workout_recovery": recoveries, "macroeconomic_history": macro,
        "reporting_date_portfolio": portfolio, "ecl_projection_cube": cube,
        "scenario_loan_ecl": scenario_ecl, "loan_level_ecl": loan_ecl,
        "pd_model_coefficients": pd_coefficients,
        "pd_development_sample": pd_development_sample,
        "lgd_development_sample": lgd_development_sample,
        "ead_development_sample": ead_development_sample,
    }
    write_outputs(tables, database_tables, cfg)
    save_models(pd_models, cfg)
    save_lgd_model(lgd_model, cfg)
    (Path(cfg["paths"]["models"]) / "ead_calibration.json").write_text(json.dumps(behavioral, indent=2), encoding="utf-8")

    if not args.skip_workbook:
        print("Creating Excel reporting workbook", flush=True)
        build_excel_report(cfg)
    total = loan_ecl["loan_ecl"].sum()
    exposure = loan_ecl["gross_exposure"].sum()
    print(f"COMPLETE: {len(loan_ecl):,} loans | exposure ${exposure:,.2f} | ECL ${total:,.2f} | coverage {total/exposure:.4%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
