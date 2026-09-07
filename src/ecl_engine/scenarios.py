from __future__ import annotations

import numpy as np
import pandas as pd


def build_scenarios(macro: pd.DataFrame, portfolio: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Create internally coherent illustrative scenarios; names/weights are modelling choices, not IFRS 9 mandates."""
    months = int(cfg["reporting"]["maximum_projection_months"])
    start = pd.Timestamp(cfg["reporting"]["reporting_date"])
    future = pd.date_range(start + pd.offsets.MonthBegin(1), periods=months, freq="MS")
    history = macro[macro["period"] <= start].dropna(subset=["unemployment_rate", "gdp_growth_yoy", "hpi_growth_yoy"])
    last = history.iloc[-1]
    long_run = history.tail(120)[["unemployment_rate", "gdp_growth_yoy", "hpi_growth_yoy"]].median()
    portfolio_rate = float(portfolio["current_interest_rate"].median())
    rows = []
    for scenario, assumptions in cfg["scenarios"].items():
        for t, period in enumerate(future, start=1):
            decay = np.exp(-t / 36)
            stress_decay = np.exp(-max(t - 12, 0) / 30)
            unemp = long_run["unemployment_rate"] + (last["unemployment_rate"] - long_run["unemployment_rate"]) * decay + assumptions["unemployment_shock_pp"] * stress_decay
            gdp = long_run["gdp_growth_yoy"] + (last["gdp_growth_yoy"] - long_run["gdp_growth_yoy"]) * decay + assumptions["gdp_growth_shock_pp"] * stress_decay
            hpi_growth = long_run["hpi_growth_yoy"] + (last["hpi_growth_yoy"] - long_run["hpi_growth_yoy"]) * decay + assumptions["hpi_growth_shock_pp"] * stress_decay
            mortgage_rate = portfolio_rate + assumptions["mortgage_rate_shock_pp"] * stress_decay
            rows.append({"scenario": scenario.title(), "scenario_weight": assumptions["weight"], "month_number": t, "period": period,
                         "unemployment_rate": max(2.0, unemp), "gdp_growth_yoy": gdp, "hpi_growth_yoy": hpi_growth,
                         "mortgage_rate": max(0.0, mortgage_rate), "scenario_is_official_forecast": False})
    paths = pd.DataFrame(rows)
    paths["hpi_factor"] = paths.groupby("scenario")["hpi_growth_yoy"].transform(
        lambda x: ((1 + x.clip(-30, 30) / 100) ** (1 / 12)).cumprod()
    )
    if abs(paths.groupby("scenario")["scenario_weight"].first().sum() - 1) > 1e-12:
        raise ValueError("Scenario weights do not sum to 100%")
    return paths


def scenario_weight_analysis(macro: pd.DataFrame, scenarios: pd.DataFrame, reporting_date=None) -> pd.DataFrame:
    """Compare configured weights with a simple historical macro-regime reference."""
    variables = ["unemployment_rate", "gdp_growth_yoy", "hpi_growth_yoy"]
    history = macro.copy()
    if reporting_date is not None:
        history = history[history["period"] <= pd.Timestamp(reporting_date)]
    history = history[variables].dropna().copy()
    means = history.mean()
    scales = history.std().replace(0, 1)
    history_z = (history - means) / scales
    first_year = scenarios[scenarios["month_number"] <= 12]
    centres = first_year.groupby("scenario")[variables].mean()
    centres_z = (centres - means) / scales
    distances = pd.DataFrame({scenario: ((history_z - centre) ** 2).sum(axis=1) for scenario, centre in centres_z.iterrows()})
    nearest = distances.idxmin(axis=1).value_counts(normalize=True)
    configured = scenarios.groupby("scenario")["scenario_weight"].first()
    rows = []
    for scenario in configured.index:
        rows.append({
            "scenario": scenario,
            "configured_weight": configured[scenario],
            "historical_analogue_frequency": float(nearest.get(scenario, 0.0)),
            "configured_component": "judgemental starting probability",
            "historical_component": "nearest historical macro-regime frequency",
        })
    return pd.DataFrame(rows)


def calibrate_scenario_weights(macro: pd.DataFrame, scenarios: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Blend configured judgement with an empirical historical-regime frequency."""
    metrics = scenario_weight_analysis(macro, scenarios, cfg["reporting"]["reporting_date"])
    share = float(cfg["scenario_weighting"]["configured_weight_share"])
    if not 0 <= share <= 1:
        raise ValueError("configured_weight_share must be between zero and one")
    metrics["selected_weight"] = (
        share * metrics["configured_weight"] + (1 - share) * metrics["historical_analogue_frequency"]
    )
    metrics["selected_weight"] = metrics["selected_weight"] / metrics["selected_weight"].sum()
    metrics["weighting_method"] = cfg["scenario_weighting"]["method"]
    selected = metrics.set_index("scenario")["selected_weight"]
    paths = scenarios.copy()
    paths["scenario_weight"] = paths["scenario"].map(selected)
    return paths, metrics
