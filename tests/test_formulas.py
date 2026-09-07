import numpy as np
import pandas as pd

from src.ecl_engine.ead import contractual_balance
from src.ecl_engine.defaults import delinquency_to_dpd


def test_delinquency_mapping():
    assert delinquency_to_dpd("00") == 0
    assert delinquency_to_dpd("1") == 30
    assert delinquency_to_dpd("3") == 90
    assert delinquency_to_dpd("RA") == 90


def test_contractual_balance_runs_down():
    months = np.array([1, 12, 120, 360])
    balance = contractual_balance(np.repeat(200000.0, 4), np.repeat(5.0, 4), np.repeat(360.0, 4), months)
    assert np.all(np.diff(balance) < 0)
    assert balance[-1] < 0.01


def test_marginal_pd_identity():
    q = pd.Series([0.01, 0.02, 0.03])
    survival = (1 - q).cumprod().shift(1, fill_value=1.0)
    mpd = survival * q
    assert np.isclose(mpd.sum(), 1 - np.prod(1 - q))
