from __future__ import annotations

import pandas as pd


STATE_ORDER = ["Current/<30", "30-59", "60-89", "Default/90+"]


def transition_matrix(panel: pd.DataFrame) -> pd.DataFrame:
    frame = panel.sort_values(["loan_id", "period"]).copy()
    frame["next_state"] = frame.groupby("loan_id")["delinquency_state"].shift(-1)
    frame["next_period"] = frame.groupby("loan_id")["period"].shift(-1)
    month_gap = (frame["next_period"].dt.year - frame["period"].dt.year) * 12 + (frame["next_period"].dt.month - frame["period"].dt.month)
    frame = frame[month_gap.eq(1)]
    counts = pd.crosstab(frame["delinquency_state"], frame["next_state"]).reindex(index=STATE_ORDER, columns=STATE_ORDER, fill_value=0)
    rates = counts.div(counts.sum(axis=1).replace(0, 1), axis=0)
    return rates.rename_axis("from_state").reset_index().melt("from_state", var_name="to_state", value_name="transition_probability")
