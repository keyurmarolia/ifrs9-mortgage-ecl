import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "database" / "ifrs9_ecl.sqlite3"


def read(sql):
    with sqlite3.connect(DB) as connection:
        return pd.read_sql_query(sql, connection)


def test_all_production_controls_pass():
    assert read("select count(*) failures from controls where status <> 'PASS'").iloc[0, 0] == 0


def test_all_three_stages_and_stage_methods_exist():
    stages = set(read("select distinct stage from loan_level_ecl")["stage"])
    assert stages == {1, 2, 3}
    stage3_methods = read("select distinct calculation_method from ecl_projection_cube where stage=3")
    assert stage3_methods["calculation_method"].str.contains("cash shortfall").all()


def test_scenario_weighting_and_portfolio_reconciliation():
    weights = read("select scenario, max(scenario_weight) weight from scenario_loan_ecl group by scenario")
    assert np.isclose(weights["weight"].sum(), 1.0)
    loans = read("select sum(loan_ecl) ecl from loan_level_ecl").iloc[0, 0]
    stages = read("select sum(ecl) ecl from stage_summary").iloc[0, 0]
    assert np.isclose(loans, stages, atol=0.01)


def test_projection_cube_has_required_audit_fields():
    required = {"loan_id", "stage", "scenario", "future_month", "conditional_pd", "survival_probability", "marginal_pd", "cumulative_pd", "ead", "projected_property_value", "projected_ltv", "lgd", "discount_factor", "period_ecl"}
    with sqlite3.connect(DB) as connection:
        columns = {row[1] for row in connection.execute("pragma table_info(ecl_projection_cube)")}
    assert required <= columns


def test_stage1_is_12_month_and_stage2_extends_beyond_12():
    horizons = read("select stage, max(future_month) max_month from ecl_projection_cube group by stage")
    mapping = dict(zip(horizons["stage"], horizons["max_month"]))
    assert mapping[1] <= 12
    assert mapping[2] > 12


def test_component_term_structures_and_scenario_views_exist():
    required_tables = {"pd_term_structure", "lgd_term_structure", "ead_term_structure", "discount_factor_term_structure"}
    required_views = {"ecl_projection_base", "ecl_projection_upside", "ecl_projection_downside", "pd_term_structure_detail", "lgd_term_structure_detail", "ead_term_structure_detail", "discount_factor_term_structure_detail"}
    with sqlite3.connect(DB) as connection:
        tables = {row[0] for row in connection.execute("select name from sqlite_master where type='table'")}
        views = {row[0] for row in connection.execute("select name from sqlite_master where type='view'")}
    assert required_tables <= tables
    assert required_views <= views


def test_model_development_outputs_exist():
    for table in ["pd_woe_iv", "pd_calibration", "pd_development_sample", "pd_macro_satellite_metrics", "lgd_development_sample", "lgd_model_metrics", "ead_development_sample", "ead_backtest_metrics", "ead_macro_sensitivity", "scenario_weight_analysis"]:
        assert read(f"select count(*) rows from {table}").iloc[0, 0] > 0


def test_monthly_ecl_equation_and_stage3_cash_shortfall():
    performing = read("""select max(abs(period_ecl - marginal_pd*lgd*ead*discount_factor)) difference
                         from ecl_projection_cube where stage in (1,2)""").iloc[0, 0]
    stage3 = read("""select max(abs(period_ecl - max(ead-discounted_expected_recovery,0))) difference
                    from ecl_projection_cube where stage=3""").iloc[0, 0]
    assert performing < 1e-8
    assert stage3 < 1e-8


def test_base_twelve_month_pd_and_stage2_horizon_reconcile():
    pd_difference = read("""select max(abs(cube_pd-current_pd_12m)) difference from (
        select p.loan_id, p.current_pd_12m, sum(c.marginal_pd) cube_pd
        from reporting_date_portfolio p join ecl_projection_cube c using(loan_id)
        where p.stage in (1,2) and c.scenario='Base' and c.future_month<=12
        group by p.loan_id, p.current_pd_12m)""").iloc[0, 0]
    horizon_mismatches = read("""select count(*) mismatches from (
        select p.loan_id, p.ecl_horizon_months, max(c.future_month) cube_horizon
        from reporting_date_portfolio p join ecl_projection_cube c using(loan_id)
        where p.stage=2 group by p.loan_id, p.ecl_horizon_months)
        where ecl_horizon_months<>cube_horizon""").iloc[0, 0]
    assert pd_difference < 1e-8
    assert horizon_mismatches == 0


def test_scenario_ordering_and_report_aliases():
    totals = read("select scenario, sum(period_ecl) ecl from ecl_projection_cube group by scenario").set_index("scenario")["ecl"]
    assert totals["Upside"] <= totals["Base"] <= totals["Downside"]
    aliases = read("select loan_id from account_level_ecl")
    assert aliases["loan_id"].str.fullmatch(r"RM\d{6}").all()
