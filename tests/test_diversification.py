import pandas as pd
import pytest
from oil_research.diversification import blend_weights
from oil_research.extension_leverage import simulate_signed


def test_opposing_exposures_net_before_costs():
    i = pd.bdate_range("2020-01-01", periods=3)
    w = blend_weights(pd.Series(1.0, i), pd.Series(-1.0, i), 0.5)
    l = simulate_signed(pd.Series(0.1, i), pd.Series(0.0001, i), pd.Series(1 / 365, i), w)
    assert l.turnover.sum() == 0
    assert l.borrow_fraction.sum() == 0
    assert l.excess_return.abs().sum() == 0


def test_blend_rejects_misalignment_and_invalid_allocation():
    a = pd.Series([1.0], index=[0])
    b = pd.Series([1.0], index=[1])
    with pytest.raises(ValueError):
        blend_weights(a, b, 0.5)
    with pytest.raises(ValueError):
        blend_weights(a, a, 1.1)
