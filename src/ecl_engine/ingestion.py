from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from .schema import ORIGINATION_COLUMNS, PERFORMANCE_COLUMNS


def _yyyymm(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series.astype("string"), format="%Y%m", errors="coerce")


def load_origination(cfg: dict) -> pd.DataFrame:
    """Business question: what risk and contractual terms existed at initial recognition?"""
    path = Path(cfg["data"]["freddie_origination"])
    frame = pd.read_csv(path, sep="|", names=ORIGINATION_COLUMNS, dtype="string", na_values=[""])
    numeric = ["classic_fico", "mortgage_insurance_percentage", "original_cltv", "original_dti",
               "original_upb", "original_ltv", "original_interest_rate", "original_loan_term",
               "number_of_borrowers"]
    for col in numeric:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame["first_payment_date"] = _yyyymm(frame["first_payment_date"])
    frame["maturity_date"] = _yyyymm(frame["maturity_date"])
    frame.loc[frame["classic_fico"].eq(9999), "classic_fico"] = np.nan
    frame.loc[frame["original_dti"].eq(999), "original_dti"] = np.nan
    frame.loc[frame["original_ltv"].eq(999), "original_ltv"] = np.nan
    frame.loc[frame["original_cltv"].eq(999), "original_cltv"] = np.nan
    frame.loc[frame["mortgage_insurance_percentage"].eq(999), "mortgage_insurance_percentage"] = np.nan
    frame["origination_date"] = frame["first_payment_date"] - pd.offsets.MonthBegin(1)
    frame["original_property_value"] = frame["original_upb"] / (frame["original_ltv"] / 100.0).replace(0, np.nan)
    frame["synthetic_field_flag"] = False
    if frame["loan_id"].duplicated().any():
        raise ValueError("Origination loan IDs are not unique")
    return frame


def select_model_loan_ids(origination: pd.DataFrame, cfg: dict) -> set[str]:
    n = min(int(cfg["data"]["model_sample_loans"]), len(origination))
    return set(origination.sample(n=n, random_state=int(cfg["project"]["random_seed"]))["loan_id"])


def load_performance(cfg: dict, loan_ids: set[str]) -> pd.DataFrame:
    """Load real Freddie Mac loan-month observations for the configured modelling sample."""
    path = Path(cfg["data"]["freddie_performance"])
    pieces: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, sep="|", names=PERFORMANCE_COLUMNS, dtype="string",
                             chunksize=int(cfg["data"]["chunk_rows"]), na_values=[""]):
        chunk = chunk[chunk["loan_id"].isin(loan_ids)]
        if not chunk.empty:
            pieces.append(chunk)
    frame = pd.concat(pieces, ignore_index=True)
    numeric = ["current_actual_upb", "loan_age", "remaining_months_to_legal_maturity",
               "current_interest_rate", "current_non_interest_bearing_upb", "mi_recoveries",
               "net_sales_proceeds", "non_mi_recoveries", "total_expenses", "actual_loss",
               "cumulative_modification_costs", "estimated_ltv", "zero_balance_removal_upb",
               "delinquent_accrued_interest", "current_interest_bearing_upb"]
    for col in numeric:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame["period"] = _yyyymm(frame["period"])
    frame["zero_balance_effective_date"] = _yyyymm(frame["zero_balance_effective_date"])
    frame = frame.dropna(subset=["period"]).sort_values(["loan_id", "period"]).drop_duplicates(["loan_id", "period"], keep="last")
    return frame.reset_index(drop=True)


def load_macro_history(cfg: dict) -> pd.DataFrame:
    """Combine official macro series and align them to when they would be available."""
    unemp = pd.read_csv(cfg["data"]["fred_unemployment"])
    unemp.columns = ["period", "unemployment_rate"]
    unemp["period"] = pd.to_datetime(unemp["period"]).dt.to_period("M").dt.to_timestamp()
    unemp["unemployment_rate"] = pd.to_numeric(unemp["unemployment_rate"], errors="coerce")
    gdp = pd.read_csv(cfg["data"]["fred_real_gdp"])
    gdp.columns = ["period", "real_gdp"]
    gdp["period"] = pd.to_datetime(gdp["period"]).dt.to_period("M").dt.to_timestamp()
    gdp["real_gdp"] = pd.to_numeric(gdp["real_gdp"], errors="coerce")
    gdp = gdp.set_index("period").resample("MS").ffill().reset_index()
    hpi_raw = pd.read_excel(cfg["data"]["fhfa_hpi_monthly"], sheet_name=0, header=3)
    month_col = hpi_raw.columns[0]
    usa_candidates = [c for c in hpi_raw.columns if "USA" in str(c) and "SA" in str(c)]
    hpi_col = usa_candidates[-1] if usa_candidates else hpi_raw.columns[-1]
    hpi = hpi_raw[[month_col, hpi_col]].copy()
    hpi.columns = ["period", "hpi_index"]
    hpi["period"] = pd.to_datetime(hpi["period"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    hpi["hpi_index"] = pd.to_numeric(hpi["hpi_index"], errors="coerce")
    macro = unemp.merge(gdp, on="period", how="outer").merge(hpi, on="period", how="outer").sort_values("period")
    macro[["unemployment_rate", "real_gdp", "hpi_index"]] = macro[["unemployment_rate", "real_gdp", "hpi_index"]].ffill()
    macro["gdp_growth_yoy"] = macro["real_gdp"].pct_change(12) * 100
    macro["hpi_growth_yoy"] = macro["hpi_index"].pct_change(12) * 100
    lags = {
        "unemployment_rate": int(cfg["macro"]["unemployment_publication_lag_months"]),
        "real_gdp": int(cfg["macro"]["gdp_publication_lag_months"]),
        "gdp_growth_yoy": int(cfg["macro"]["gdp_publication_lag_months"]),
        "hpi_index": int(cfg["macro"]["hpi_publication_lag_months"]),
        "hpi_growth_yoy": int(cfg["macro"]["hpi_publication_lag_months"]),
    }
    for column, months in lags.items():
        macro[column] = macro[column].shift(months)
    if bool(cfg["macro"].get("use_reporting_date_cutoff", True)):
        macro = macro[macro["period"] <= pd.Timestamp(cfg["reporting"]["reporting_date"])].copy()
    # Freddie sample has fixed-rate mortgages; PMMS is absent, so portfolio weighted note rate is used later and disclosed.
    macro["mortgage_rate"] = np.nan
    macro["source_status"] = (
        "official observations aligned with configured publication lags; "
        "GDP carried forward between quarterly releases; revised-history limitation disclosed"
    )
    return macro.dropna(subset=["period"]).reset_index(drop=True)
