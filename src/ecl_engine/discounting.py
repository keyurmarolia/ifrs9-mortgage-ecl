from __future__ import annotations

import pandas as pd


def add_discount_factors(cube: pd.DataFrame) -> pd.DataFrame:
    frame = cube.copy()
    monthly_eir = frame["current_interest_rate"].fillna(0) / 100 / 12
    frame["discount_factor"] = 1 / ((1 + monthly_eir) ** frame["future_month"])
    frame["discount_rate_method"] = "loan-specific current contractual mortgage rate as EIR approximation"
    return frame
